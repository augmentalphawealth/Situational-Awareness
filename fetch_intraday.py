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

# HOLIDAY GUARD: NSE:NIFTY 50
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
    print(f"⚠️ {cache_file} not found. Attempting to build it from the master database...")
    master_file = "nse_6yr_historical.parquet"
    if os.path.exists(master_file):
        df_master = pd.read_parquet(master_file)
        df_master['Date'] = pd.to_datetime(df_master['Date'])
        cutoff_date = df_master['Date'].max() - pd.Timedelta(days=450)
        df_cache = df_master[df_master['Date'] >= cutoff_date]
        df_cache.to_parquet(cache_file, index=False)
        print("✅ Cache built successfully.")
    else:
        print(f"❌ Master database '{master_file}' is also missing. Cannot proceed.")
        sys.exit(1)

df_hist = pd.read_parquet(cache_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
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
            if "429" in str(e) or "403" in str(e):
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
    time.sleep(1.1)  

if not new_rows:
    print("❌ No data fetched. Access token might be invalid.")
    sys.exit(1)

df_live = pd.DataFrame(new_rows)
df_live['Date'] = pd.to_datetime(df_live['Date']).dt.normalize()
df_hist = df_hist[df_hist['Date'] != today_dt]
df = pd.concat([df_hist, df_live], ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)

df['Daily_Turnover'] = df['Close'] * df['Volume']
df['Turnover_45d_Avg'] = df.groupby('Symbol')['Daily_Turnover'].transform(lambda x: x.shift(1).rolling(window=45, min_periods=10).mean())
df['Cap_Rank'] = df.groupby('Date')['Turnover_45d_Avg'].rank(ascending=False, method='min')

conditions = [(df['Cap_Rank'] <= 100), (df['Cap_Rank'] > 100) & (df['Cap_Rank'] <= 250), (df['Cap_Rank'] > 250) & (df['Cap_Rank'] <= 500)]
df['Liquidity_Category'] = np.select(conditions, ['Top 100 Liq', 'Mid 150 Liq', 'Lower 250 Liq'], default='Micro Liq')

df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=20).mean())
df['EMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=50).mean())
df['EMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=200).mean())
df['Daily_Pct'] = df.groupby('Symbol')['Close'].pct_change() * 100
df['Pct_1M'] = df.groupby('Symbol')['Close'].pct_change(periods=21) * 100

traded_today = df['Volume'] > 0
df['Gainer'] = (df['Daily_Pct'] > 0) & traded_today
df['Loser'] = (df['Daily_Pct'] < 0) & traded_today
df['Rolling_52W_High'] = df.groupby('Symbol')['High'].transform(lambda x: x.shift(1).rolling(window=252, min_periods=200).max())
df['Rolling_52W_Low'] = df.groupby('Symbol')['Low'].transform(lambda x: x.shift(1).rolling(window=252, min_periods=200).min())

df['Above_20_EMA'] = df['Close'] > df['EMA_20']
df['Above_50_EMA'] = df['Close'] > df['EMA_50']
df['Above_200_EMA'] = df['Close'] > df['EMA_200']
df['Valid_20_EMA'] = df['EMA_20'].notna()
df['Valid_50_EMA'] = df['EMA_50'].notna()
df['Valid_200_EMA'] = df['EMA_200'].notna()
df['Up_4_Pct'] = (df['Daily_Pct'] >= 4.0) & traded_today
df['Down_4_Pct'] = (df['Daily_Pct'] <= -4.0) & traded_today
df['Up_25_1M'] = df['Pct_1M'] >= 25.0
df['Down_25_1M'] = df['Pct_1M'] <= -25.0
df['New_52W_High'] = (df['Close'] >= df['Rolling_52W_High']) & traded_today
df['New_52W_Low'] = (df['Close'] <= df['Rolling_52W_Low']) & traded_today

df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
df['ATR_14'] = df.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
df['Vol_20D_Avg'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
df['Max_20D_Prior'] = df.groupby('Symbol')['High'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())
df['20D_High'] = df['Close'] > df['Max_20D_Prior']
df['VCP_Tightness'] = (df['ATR_14'] / df['Close']) < 0.04

# Intraday Volume Projection
market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
elapsed_minutes = max(1, min(375, int((now_ist - market_open).total_seconds() / 60)))
df['Projected_Volume'] = np.where(df['Date'] == today_dt, df['Volume'] * (375 / elapsed_minutes), df['Volume'])
df['Volume_Surge'] = df['Projected_Volume'] > (df['Vol_20D_Avg'] * 1.5)

prior_tight = df.groupby('Symbol')['VCP_Tightness'].shift(1).astype(float).fillna(0.0).astype(bool)
df['Is_Breakout'] = df['20D_High'] & df['Volume_Surge'] & prior_tight

df['Is_Breakout_3d_ago'] = df.groupby('Symbol')['Is_Breakout'].shift(3).astype(float).fillna(0.0).astype(bool)
df['Close_3d_ago'] = df.groupby('Symbol')['Close'].shift(3)
df['Follow_Through_Win'] = df['Is_Breakout_3d_ago'] & (df['Close'] > df['Close_3d_ago'])

df['Up_Volume'] = np.where(df['Gainer'], df['Daily_Turnover'], 0)
df['Down_Volume'] = np.where(df['Loser'], df['Daily_Turnover'], 0)

overall_breadth = df.groupby('Date').agg(
    Total_Universe=('Symbol', 'count'), Advances=('Gainer', 'sum'), Declines=('Loser', 'sum')
).reset_index()

# ONLY update the live intraday breadth file. Do NOT touch the 6-year history.
intra_df = pd.DataFrame([{"Time": now_ist.strftime("%H:%M"), "Advances": overall_breadth.iloc[-1]['Advances'], "Declines": overall_breadth.iloc[-1]['Declines'], "Date": now_ist.strftime("%Y-%m-%d")}])
intra_df.to_csv("live_intraday_breadth.csv", index=False)

with open("last_sync.txt", "w") as f: f.write(f"Today, {now_ist.strftime('%I:%M %p')} IST")
