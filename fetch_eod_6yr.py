import pandas as pd
import datetime
import os
import sys
import time
from kiteconnect import KiteConnect

api_key = os.environ.get("KITE_API_KEY")
access_token = os.environ.get("KITE_ACCESS_TOKEN")

print("Connecting to Zerodha API for EOD Sync & Healing...")
try:
    if not api_key or not access_token:
        print("❌ KITE_API_KEY or KITE_ACCESS_TOKEN missing.")
        sys.exit(1)
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
except Exception as e:
    print("❌ Error connecting:", e)
    sys.exit(1)

today_dt = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d")).normalize()

try:
    test_quote = kite.quote(["NSE:NIFTY 50"])
    if "NSE:NIFTY 50" in test_quote:
        last_time = test_quote["NSE:NIFTY 50"].get("last_trade_time")
        if last_time and last_time.date() != today_dt.date():
            print(f"🛑 MARKET HOLIDAY GUARD: NIFTY 50 last trade on {last_time.date()}. Aborting sync.")
            sys.exit(0)
except Exception as e:
    print("⚠️ Holiday guard check failed, proceeding with caution.", e)

parquet_file = "nse_6yr_historical.parquet"
tmp_parquet = "nse_6yr_historical.tmp.parquet"
if not os.path.exists(parquet_file):
    print(f"❌ Error: Master database '{parquet_file}' not found.")
    sys.exit(1)

df_hist = pd.read_parquet(parquet_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
unique_symbols = df_hist['Symbol'].unique()

kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunks = [kite_symbols[i:i + 200] for i in range(0, len(kite_symbols), 200)]

new_rows = []
heal_queue = []

# Fetch Today's Quotes & Check for Corporate Action Gaps
print("Fetching daily quotes and scanning for 2% corporate action gaps...")
for chunk in chunks:
    for attempt in range(5):
        try:
            res = kite.quote(chunk)
            if res:
                for symbol_nse, data in res.items():
                    sym = symbol_nse.replace("NSE:", "")
                    
                    # 2% Gap Checker
                    try:
                        kite_prev_close = data['ohlc']['close']
                        db_prev_close = df_hist[df_hist['Symbol'] == sym]['Close'].iloc[-1]
                        if pd.notna(kite_prev_close) and pd.notna(db_prev_close) and db_prev_close > 0:
                            if abs(db_prev_close - kite_prev_close) / db_prev_close > 0.02:
                                heal_queue.append({'symbol': sym, 'token': data['instrument_token']})
                    except:
                        pass
                        
                    new_rows.append({
                        "Date": today_dt, "Open": data['ohlc']['open'], "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'], "Close": data['last_price'], "Volume": data['volume'],
                        "Symbol": sym
                    })
                break
        except Exception as e:
            if "429" in str(e) or "403" in str(e):
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
    time.sleep(1.1)

if not new_rows:
    print("❌ No data fetched. Access token might be invalid.")
    sys.exit(1)

# Corporate Action Healer
if heal_queue:
    print(f"⚠️ Suspected Corporate Actions Detected for {len(heal_queue)} symbols. Initiating Atomic Healer...")
    healed_data = []
    symbols_healed = set()
    
    yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    for item in heal_queue:
        sym = item['symbol']
        stock_data = []
        is_valid = True
        
        for chunk_dates in [("2018-04-01", "2022-03-31"), ("2022-04-01", yesterday_str)]:
            for attempt in range(5):
                try:
                    res = kite.historical_data(item['token'], chunk_dates[0], chunk_dates[1], "day")
                    if res: stock_data.extend(res)
                    break
                except Exception as e:
                    if "429" in str(e) or "403" in str(e): time.sleep(2 ** attempt)
                    else: time.sleep(1)
            time.sleep(0.4)
            
        if stock_data:
            df_temp = pd.DataFrame(stock_data).rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            df_temp['Symbol'] = sym
            df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.normalize()
            
            # Strict Validation Guard
            if len(df_temp) > 10 and (df_temp['Close'] > 0).all() and (df_temp['High'] >= df_temp['Low']).all():
                healed_data.append(df_temp)
                symbols_healed.add(sym)
                print(f"   [+] Successfully healed history for {sym}")
            else:
                print(f"   [-] Validation failed for {sym}, skipping heal.")
    
    # Atomic Swap in Memory
    if symbols_healed:
        df_hist = df_hist[~df_hist['Symbol'].isin(symbols_healed)]
        df_healed = pd.concat(healed_data, ignore_index=True)
        df_hist = pd.concat([df_hist, df_healed], ignore_index=True)

if len(new_rows) < 0.98 * len(unique_symbols):
    print(f"❌ Completeness Guard Failed! Fetched {len(new_rows)}/{len(unique_symbols)}. Aborting save.")
    sys.exit(1)

new_eod_df = pd.DataFrame(new_rows)
df_hist = df_hist[df_hist['Date'] != today_dt]
combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
combined = combined.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")

combined.to_parquet(tmp_parquet, index=False)
os.replace(tmp_parquet, parquet_file)
print("✅ EOD DB saved. Triggering Math...")
os.system("python analyze_6yr_data.py")
