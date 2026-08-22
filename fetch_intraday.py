import pandas as pd
import numpy as np
import datetime
import os
import sys
import time
from kiteconnect import KiteConnect

api_key = os.environ.get("KITE_API_KEY")
access_token = os.environ.get("KITE_ACCESS_TOKEN")

print("Connecting to Zerodha API for Intraday Snapshot...")
try:
    if not api_key or not access_token:
        print("❌ KITE_API_KEY or KITE_ACCESS_TOKEN missing.")
        sys.exit(1)
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
except Exception as e:
    print("❌ Error connecting:", e)
    sys.exit(1)

ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
now_ist = datetime.datetime.now(ist_offset)
today_dt = pd.to_datetime(now_ist.strftime("%Y-%m-%d")).normalize()

try:
    test_quote = kite.quote(["NSE:NIFTY 50"])
    if "NSE:NIFTY 50" in test_quote:
        last_time = test_quote["NSE:NIFTY 50"].get("last_trade_time")
        if last_time and last_time.date() != today_dt.date():
            print(f"🛑 MARKET HOLIDAY GUARD: NIFTY 50 last trade on {last_time.date()}. Aborting sync.")
            sys.exit(0)
except Exception as e:
    print("⚠️ Holiday guard check failed, proceeding with caution.", e)

cache_file = "trailing_cache.parquet"
if not os.path.exists(cache_file):
    print("❌ Cache missing. Wait for EOD sync to rebuild.")
    sys.exit(1)

df_hist = pd.read_parquet(cache_file)
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
                        "Date": today_dt, "Symbol": symbol.replace("NSE:", ""),
                        "Open": data['ohlc']['open'], "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'], "Close": data['last_price'], "Volume": data['volume'] 
                    })
                break
        except Exception as e:
            if "429" in str(e) or "403" in str(e): time.sleep(2 ** attempt)
            else: time.sleep(1)
    time.sleep(1.1)  

if not new_rows:
    sys.exit(1)

df_live = pd.DataFrame(new_rows)
df_live['Date'] = pd.to_datetime(df_live['Date']).dt.normalize()
df_hist = df_hist[df_hist['Date'] != today_dt]
df = pd.concat([df_hist, df_live], ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)

# Intraday VIP Gating
df['History_Days'] = df.groupby('Symbol').cumcount() + 1
df['Prior_History_Days'] = df['History_Days'] - 1
df['Daily_Turnover'] = df['Close'] * df['Volume']
df['Prior_Turnover_20D_Avg'] = df.groupby('Symbol')['Daily_Turnover'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())

mature_valid = (df['Prior_History_Days'] >= 20) & (df['Prior_Turnover_20D_Avg'] >= 50_000_000)
new_valid = (df['Prior_History_Days'] >= 1) & (df['Prior_History_Days'] < 20)
df['Active_Universe'] = (mature_valid | new_valid) & (df['Volume'] > 0)

df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
df['Daily_Pct'] = df.groupby('Symbol')['Close'].pct_change() * 100

df['Gainer'] = df['Active_Universe'] & (df['Daily_Pct'] > 0)
df['Loser'] = df['Active_Universe'] & (df['Daily_Pct'] < 0)

overall_breadth = df.groupby('Date').agg(
    Total_Universe=('Active_Universe', 'sum'), Advances=('Gainer', 'sum'), Declines=('Loser', 'sum')
).reset_index()

intra_adv = int(overall_breadth.iloc[-1]['Advances'])
intra_dec = int(overall_breadth.iloc[-1]['Declines'])

# 🔥 NEW FLATLINE/HOLIDAY GUARD TO PREVENT THE 0/0 BUG
if intra_adv == 0 and intra_dec == 0:
    print("🛑 ZERO VOLUME / HOLIDAY DETECTED: 0 Advances and 0 Declines. Aborting file save.")
    sys.exit(0)

intra_df = pd.DataFrame([{"Time": now_ist.strftime("%H:%M"), "Advances": intra_adv, "Declines": intra_dec, "Date": now_ist.strftime("%Y-%m-%d")}])
intra_df.to_csv("live_intraday_breadth.csv", index=False)

with open("last_sync.txt", "w") as f: f.write(f"Today, {now_ist.strftime('%I:%M %p')} IST")
