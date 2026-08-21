import pandas as pd
import datetime
import os
import sys
import time
from kiteconnect import KiteConnect

api_key = os.environ.get("KITE_API_KEY")
access_token = os.environ.get("KITE_ACCESS_TOKEN")

print("Connecting to Zerodha API for EOD Sync...")
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

# HOLIDAY GUARD: Proxy switched to NIFTY 50
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
for chunk in chunks:
    for attempt in range(5):
        try:
            res = kite.quote(chunk)
            if res:
                for symbol, data in res.items():
                    new_rows.append({
                        "Date": today_dt, "Open": data['ohlc']['open'], "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'], "Close": data['last_price'], "Volume": data['volume'],
                        "Symbol": symbol.replace("NSE:", "")
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

# 98% Completeness Guard
if len(new_rows) < 0.98 * len(unique_symbols):
    print(f"❌ Completeness Guard Failed! Fetched {len(new_rows)}/{len(unique_symbols)}. Aborting save.")
    with open("fetch_errors.log", "a") as f:
        f.write(f"[{datetime.datetime.now()}] EOD fetch failed 98% threshold. Got {len(new_rows)} symbols.\n")
    sys.exit(1)

new_eod_df = pd.DataFrame(new_rows)
df_hist = df_hist[df_hist['Date'] != today_dt]
combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
combined = combined.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")

combined.to_parquet(tmp_parquet, index=False)
os.replace(tmp_parquet, parquet_file)
print("✅ EOD DB saved. Triggering Math...")
os.system("python analyze_6yr_data.py")
