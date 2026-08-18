import pandas as pd
import numpy as np
import requests
import datetime
import pyotp
import os
import sys
import time
import re
import logzero
import logging
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs

logzero.logger.setLevel(logging.FATAL)

api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")

print("Logging in to Zerodha for Intraday Engine Snapshot...")
try:
    kite = KiteConnect(api_key=api_key)
    session = requests.Session()
    
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    
    login_res = session.post("https://kite.zerodha.com/api/login", data={"user_id": user_id, "password": password}).json()
    if login_res.get("status") != "success":
        raise Exception(f"Zerodha Login Failed: {login_res}")
    request_id = login_res["data"]["request_id"]
    
    totp_token = pyotp.TOTP(totp_secret).now()
    twofa_res = session.post("https://kite.zerodha.com/api/twofa", data={
        "user_id": user_id, 
        "request_id": request_id, 
        "twofa_value": totp_token
    }).json()
    
    if twofa_res.get("status") != "success":
        raise Exception(f"Zerodha 2FA Failed: {twofa_res}")
    
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&skip_session=true"
    redirect_res = session.get(login_url, allow_redirects=True)
    
    # Handle Authorization Consent Page if hit
    if "connect/authorize" in redirect_res.url:
        inputs = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', redirect_res.text)
        form_data = {name: val for name, val in inputs}
        if not form_data:
            parsed_url = urlparse(redirect_res.url)
            query_params = parse_qs(parsed_url.query)
            form_data = {
                "api_key": api_key,
                "sess_id": query_params.get("sess_id", [""])[0]
            }
        redirect_res = session.post(
            "https://kite.zerodha.com/connect/authorize",
            data=form_data,
            allow_redirects=True
        )
    
    request_token = None
    for resp in redirect_res.history + [redirect_res]:
        qs = parse_qs(urlparse(resp.url).query)
        if "request_token" in qs:
            request_token = qs["request_token"][0]
            break
            
    if not request_token:
        raise Exception(f"Request token missing. Final URL: {redirect_res.url}")
        
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print("✅ Zerodha Authentication Successful.")
except Exception as e:
    print("❌ Error logging in:", e)
    sys.exit(1)

# 1. Load Cache
cache_file = "trailing_cache.parquet"
if not os.path.exists(cache_file):
    print("❌ Trailing cache not found. Please run the EOD sync once to generate it.")
    sys.exit(1)

print("Loading lightweight lookback cache...")
df_hist = pd.read_parquet(cache_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
unique_symbols = df_hist['Symbol'].unique()

# 2. Fetch Live Quotes via Kite Chunking
kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunk_size = 200 
chunks = [kite_symbols[i:i + chunk_size] for i in range(0, len(kite_symbols), chunk_size)]

ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
now_ist = datetime.datetime.now(ist_offset)
today_str = now_ist.strftime("%Y-%m-%d")
today_dt = pd.to_datetime(today_str)
time_now_str = now_ist.strftime("%H:%M")

new_rows = []
print(f"Fetching live intraday quotes for Engine calculation...")
for chunk in chunks:
    for attempt in range(5): 
        try:
            res = kite.quote(chunk)
            if res:
                for symbol, data in res.items():
                    clean_symbol = symbol.replace("NSE:", "")
                    new_rows.append({
                        "Date": today_dt,
                        "Symbol": clean_symbol,
                        "Open": data['ohlc']['open'],
                        "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'],
                        "Close": data['last_price'],
                        "Volume": data['volume'] 
                    })
                break
        except Exception:
            time.sleep(2)
    time.sleep(0.4) 

if not new_rows:
    print("⚠️ No data fetched. API might be down.")
    sys.exit(1)

df_live = pd.DataFrame(new_rows)
df_live['Date'] = pd.to_datetime(df_live['Date'])
df_hist = df_hist[df_hist['Date'] != today_dt]

# 3. Concatenate and Calculate Complete Engine Math
print("Calculating Real-Time EMAs and Engine Scores...")
df = pd.concat([df_hist, df_live], ignore_index=True)
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

df['Daily_Turnover'] = df['Close'] * df['Volume']
df['Turnover_45d_Avg'] = df.groupby('Symbol')['Daily_Turnover'].transform(lambda x: x.rolling(window=45, min_periods=10).mean())
df['Cap_Rank'] = df.groupby('Date')['Turnover_45d_Avg'].rank(ascending=False, method='min')

conditions = [
    (df['Cap_Rank'] <= 100), (df['Cap_Rank'] > 100) & (df['Cap_Rank'] <= 250), (df['Cap_Rank'] > 250) & (df['Cap_Rank'] <= 500)
]
choices = ['Top 100 Liq', 'Mid 150 Liq', 'Lower 250 Liq']
df['Liquidity_Category'] = np.select(conditions, choices, default='Micro Liq')

df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=20).mean())
df['EMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=50).mean())
df['EMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=200).mean())

df['Daily_Pct'] = df.groupby('Symbol')['Close'].pct_change() * 100
df['Daily_Pct'] = df['Daily_Pct'].replace([np.inf, -np.inf], 0).fillna(0) 
df['Pct_1M'] = df.groupby('Symbol')['Close'].pct_change(periods=21) * 100

traded_today = df['Volume'] > 0
df['Gainer'] = (df['Daily_Pct'] > 0) & traded_today
df['Loser'] = (df['Daily_Pct'] < 0) & traded_today

df['Rolling_52W_High'] = df.groupby('Symbol')['High'].transform(lambda x: x.rolling(window=252, min_periods=200).max())
df['Rolling_52W_Low'] = df.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window=252, min_periods=200).min())

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

df['New_52W_High'] = df['Close'] >= df['Rolling_52W_High']
df['New_52W_Low'] = df['Close'] <= df['Rolling_52W_Low']

df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
df['ATR_14'] = df.groupby('Symbol')['TR'].transform(lambda x: x.rolling(14, min_periods=5).mean())
df['Vol_20D_Avg'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

df['20D_High'] = df.groupby('Symbol')['Close'].transform(lambda x: x == x.rolling(20, min_periods=20).max())
df['VCP_Tightness'] = (df['ATR_14'] / df['Close']) < 0.04
df['Volume_Surge'] = df['Volume'] > (df['Vol_20D_Avg'] * 1.5)

df['Is_Breakout'] = df['20D_High'] & df['Volume_Surge'] & df.groupby('Symbol')['VCP_Tightness'].shift(1)
df['Is_Breakout_3d_ago'] = df.groupby('Symbol')['Is_Breakout'].shift(3)
df['Close_3d_ago'] = df.groupby('Symbol')['Close'].shift(3)
df['Follow_Through_Win'] = (df['Is_Breakout_3d_ago'] == True) & (df['Close'] > df['Close_3d_ago'])

df['Up_Volume'] = np.where(df['Gainer'], df['Daily_Turnover'], 0)
df['Down_Volume'] = np.where(df['Loser'], df['Daily_Turnover'], 0)

# 4. Aggregations
def get_liq_breadth(data, liq_name, prefix):
    liq_df = data[data['Liquidity_Category'] == liq_name]
    aggregated = liq_df.groupby('Date').agg(
        Valid_20=('Valid_20_EMA', 'sum'), Valid_50=('Valid_50_EMA', 'sum'), Valid_200=('Valid_200_EMA', 'sum'),
        Above_20=('Above_20_EMA', 'sum'), Above_50=('Above_50_EMA', 'sum'), Above_200=('Above_200_EMA', 'sum')
    ).reset_index()
    aggregated[f'{prefix}_Pct_20_EMA'] = (aggregated['Above_20'] / aggregated['Valid_20'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_50_EMA'] = (aggregated['Above_50'] / aggregated['Valid_50'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_200_EMA'] = (aggregated['Above_200'] / aggregated['Valid_200'].replace(0, np.nan)) * 100
    return aggregated[['Date', f'{prefix}_Pct_20_EMA', f'{prefix}_Pct_50_EMA', f'{prefix}_Pct_200_EMA']]

large_breadth = get_liq_breadth(df, 'Top 100 Liq', 'Large')
mid_breadth = get_liq_breadth(df, 'Mid 150 Liq', 'Mid')
small_breadth = get_liq_breadth(df, 'Lower 250 Liq', 'Small')
micro_breadth = get_liq_breadth(df, 'Micro Liq', 'Micro')

overall_breadth = df.groupby('Date').agg(
    Total_Universe=('Symbol', 'count'),
    Valid_20=('Valid_20_EMA', 'sum'), Valid_50=('Valid_50_EMA', 'sum'), Valid_200=('Valid_200_EMA', 'sum'),
    Advances=('Gainer', 'sum'), Declines=('Loser', 'sum'),
    Above_20_EMA=('Above_20_EMA', 'sum'), Above_50_EMA=('Above_50_EMA', 'sum'), Above_200_EMA=('Above_200_EMA', 'sum'),
    Up_4_Count=('Up_4_Pct', 'sum'), Down_4_Count=('Down_4_Pct', 'sum'),
    Up_25_1M_Count=('Up_25_1M', 'sum'), Down_25_1M_Count=('Down_25_1M', 'sum'),
    New_52W_Highs=('New_52W_High', 'sum'), New_52W_Lows=('New_52W_Low', 'sum'),
    Total_Up_Volume=('Up_Volume', 'sum'), Total_Down_Volume=('Down_Volume', 'sum'),
    T3_Breakouts=('Is_Breakout_3d_ago', 'sum'), T3_Wins=('Follow_Through_Win', 'sum')
).reset_index()

overall_breadth['Pct_Above_20_EMA'] = (overall_breadth['Above_20_EMA'] / overall_breadth['Valid_20'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_50_EMA'] = (overall_breadth['Above_50_EMA'] / overall_breadth['Valid_50'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_200_EMA'] = (overall_breadth['Above_200_EMA'] / overall_breadth['Valid_200'].replace(0, np.nan)) * 100

overall_breadth['Rolling_3D_Up_4'] = overall_breadth['Up_4_Count'].rolling(window=3).sum()
overall_breadth['Rolling_3D_Down_4'] = overall_breadth['Down_4_Count'].rolling(window=3).sum()
overall_breadth['Net_52W_High_Low'] = overall_breadth['New_52W_Highs'] - overall_breadth['New_52W_Lows']
overall_breadth['Volume_Ratio'] = (overall_breadth['Total_Up_Volume'] / (overall_breadth['Total_Down_Volume'] + 1)).clip(upper=15.0)

final_summary = overall_breadth.merge(large_breadth, on='Date', how='left')
final_summary = final_summary.merge(mid_breadth, on='Date', how='left')
final_summary = final_summary.merge(small_breadth, on='Date', how='left')
final_summary = final_summary.merge(micro_breadth, on='Date', how='left')
final_summary = final_summary.drop(columns=['Valid_20', 'Valid_50', 'Valid_200'])

# 5. Overwrite the historical dashboard file with live snapshot
output_csv = "historical_breadth_regime_6yr.csv"
final_summary.to_csv(output_csv, index=False)
print("✅ Dashboard database overwritten with LIVE INTRADAY engine snapshot.")

# Live Counter for UI
live_advances = final_summary.iloc[-1]['Advances']
live_declines = final_summary.iloc[-1]['Declines']
live_total = final_summary.iloc[-1]['Total_Universe']
adv_pct_now = round((live_advances / (live_advances + live_declines) * 100), 1)

intraday_log_file = "live_intraday_breadth.csv"
intra_df = pd.DataFrame([{"Time": time_now_str, "Adv_Pct": adv_pct_now, "Advances": live_advances, "Declines": live_declines, "Date": today_str, "Total_Universe": live_total}])
intra_df.to_csv(intraday_log_file, index=False)

with open("last_sync.txt", "w") as f:
    f.write(f"Today, {now_ist.strftime('%I:%M %p')} IST")

print(f"✅ Real Intraday Sync Complete: {time_now_str} IST")
