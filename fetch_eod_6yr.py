import pandas as pd
import datetime
import os
import sys
import time
from zoneinfo import ZoneInfo
from kiteconnect import KiteConnect

# --- HARD GUARDS (IST AWARE) ---
IST = ZoneInfo("Asia/Kolkata")
now_ist = datetime.datetime.now(IST)

if now_ist.weekday() >= 5:  # 5=Saturday, 6=Sunday
    print(f"🛑 WEEKEND GUARD: {now_ist.strftime('%A, %d %b %Y')} is not a trading day. Aborting.")
    sys.exit(0)

if now_ist.hour < 7:
    print(f"🛑 TOO EARLY: {now_ist.strftime('%H:%M')} IST. Kite refreshes the previous-close field "
          f"around 6:30 AM on trading days. Aborting to prevent stale data.")
    sys.exit(0)

today_dt_ist = pd.to_datetime(now_ist.strftime("%Y-%m-%d")).normalize()

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

# --- FAIL-CLOSED HOLIDAY GUARD ---
try:
    test_quote = kite.quote(["NSE:NIFTY 50"])
    if "NSE:NIFTY 50" not in test_quote:
        print("🛑 Could not retrieve NIFTY 50 quote to verify trading day. Aborting for safety.")
        sys.exit(1)
        
    nifty_data = test_quote["NSE:NIFTY 50"]
    ltt = nifty_data.get("last_trade_time") or nifty_data.get("timestamp")
    
    if not ltt:
        print("🛑 Kite API returned no timestamp for NIFTY 50. Aborting.")
        sys.exit(1)
        
    true_latest_market_date = pd.to_datetime(ltt).tz_localize(None).normalize()
    
    if true_latest_market_date < today_dt_ist:
        print(f"🛑 MARKET HOLIDAY GUARD: NIFTY 50 last trade on {true_latest_market_date.date()}. Aborting sync.")
        sys.exit(0)

except SystemExit:
    raise
except Exception as e:
    print(f"🛑 Holiday guard check failed — aborting rather than proceeding on an unverified day. Error: {e}")
    sys.exit(1)

# --- DATABASE SYNC VERIFICATION ---
parquet_file = "nse_6yr_historical.parquet"
tmp_parquet = "nse_6yr_historical.tmp.parquet"
if not os.path.exists(parquet_file):
    print(f"❌ Error: Master database '{parquet_file}' not found.")
    sys.exit(1)

df_hist = pd.read_parquet(parquet_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
db_max_date = df_hist['Date'].max()

if db_max_date >= true_latest_market_date:
    print(f"🛑 UP-TO-DATE: Database (Last: {db_max_date.date()}) is fully synced with Market (Last: {true_latest_market_date.date()}). Aborting sync.")
    sys.exit(0)

# --- ESTABLISH TRUE PREVIOUS DATE FOR GAP CHECKER ---
try:
    nifty_token = nifty_data["instrument_token"]
    start_date = (true_latest_market_date - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    end_date = true_latest_market_date.strftime("%Y-%m-%d")
    nifty_hist = kite.historical_data(nifty_token, start_date, end_date, "day")
    
    valid_dates = [pd.to_datetime(c['date']).tz_localize(None).normalize() for c in nifty_hist]
    past_dates = [d for d in valid_dates if d < true_latest_market_date]
    target_prev_date = past_dates[-1] if past_dates else None
    print(f"📅 Quote Date: {true_latest_market_date.date()} | OHLC Previous Date Mapping: {target_prev_date.date() if target_prev_date else 'Unknown'}")
except Exception as e:
    print(f"⚠️ Could not fetch market calendar: {e}")
    target_prev_date = None

unique_symbols = df_hist['Symbol'].unique()
kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunks = [kite_symbols[i:i + 200] for i in range(0, len(kite_symbols), 200)]

new_rows = []
heal_queue = []

print("Fetching daily quotes and scanning for 2% corporate action gaps...")
for chunk in chunks:
    for attempt in range(5):
        try:
            res = kite.quote(chunk)
            if res:
                for symbol_nse, data in res.items():
                    sym = symbol_nse.replace("NSE:", "")
                    
                    # SAFE 2% Gap Checker with Strict Date Alignment
                    try:
                        sym_hist = df_hist[df_hist['Symbol'] == sym]
                        if not sym_hist.empty and target_prev_date is not None:
                            db_last_date = sym_hist['Date'].iloc[-1]
                            
                            # ONLY run the gap check if the database is perfectly up-to-date for this symbol
                            if db_last_date == target_prev_date:
                                db_prev_close = sym_hist['Close'].iloc[-1]
                                kite_prev_close = data.get('ohlc', {}).get('close', 0)
                                
                                if pd.notna(kite_prev_close) and pd.notna(db_prev_close) and db_prev_close > 0 and kite_prev_close > 0:
                                    if abs(db_prev_close - kite_prev_close) / db_prev_close > 0.02:
                                        heal_queue.append({'symbol': sym, 'token': data['instrument_token']})
                    except Exception:
                        pass
                        
                    new_rows.append({
                        "Date": true_latest_market_date, # Uses True API Date, not Server Clock
                        "Open": data.get('ohlc', {}).get('open', 0), 
                        "High": data.get('ohlc', {}).get('high', 0),
                        "Low": data.get('ohlc', {}).get('low', 0), 
                        "Close": data.get('last_price', 0), 
                        "Volume": data.get('volume', 0),
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

# --- CLAUDE'S CIRCUIT BREAKER ---
CIRCUIT_BREAKER_LIMIT = max(50, int(0.02 * len(unique_symbols)))
if len(heal_queue) > CIRCUIT_BREAKER_LIMIT:
    print(f"🛑 CIRCUIT BREAKER: {len(heal_queue)} symbols flagged simultaneously "
          f"(limit {CIRCUIT_BREAKER_LIMIT}). That's a sign the previous-close comparison "
          f"itself is unreliable right now, not {len(heal_queue)} real corporate actions. "
          f"Skipping the heal this run — investigate before rerunning.")
    heal_queue = []

# --- CORPORATE ACTION HEALER ---
if heal_queue:
    print(f"⚠️ Suspected Corporate Actions Detected for {len(heal_queue)} symbols. Initiating Atomic Healer...")
    healed_data = []
    symbols_healed = set()
    
    # Safely fetch history up to the True Market Date
    fetch_end_str = true_latest_market_date.strftime("%Y-%m-%d")
    
    for item in heal_queue:
        sym = item['symbol']
        stock_data = []
        
        for chunk_dates in [("2018-04-01", "2022-03-31"), ("2022-04-01", fetch_end_str)]:
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
            
            # Validation Guard
            if not df_temp.empty and (df_temp['Close'] > 0).all() and (df_temp['High'] >= df_temp['Low']).all():
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

# --- FINAL COMPLETENESS & SAVE ---
if len(new_rows) < 0.98 * len(unique_symbols):
    print(f"❌ Completeness Guard Failed! Fetched {len(new_rows)}/{len(unique_symbols)}. Aborting save.")
    sys.exit(1)

new_eod_df = pd.DataFrame(new_rows)
# Remove any existing rows for the exact market date before appending to prevent duplicates
df_hist = df_hist[df_hist['Date'] != true_latest_market_date]
combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
combined = combined.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")

combined.to_parquet(tmp_parquet, index=False)
os.replace(tmp_parquet, parquet_file)
print("✅ EOD DB saved. Triggering Math...")

exit_code = os.system("python analyze_6yr_data.py")
if exit_code != 0:
    print("❌ analyze_6yr_data.py failed!")
    sys.exit(1)
