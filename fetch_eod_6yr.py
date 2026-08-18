import pandas as pd
import requests
import datetime
import os
import sys
import time
import pyotp
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs

api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")

print("Logging in to Zerodha for EOD Batch Sync...")
try:
    kite = KiteConnect(api_key=api_key)
    session = requests.Session()
    login_res = session.post("https://kite.zerodha.com/api/login", data={"user_id": user_id, "password": password}).json()
    request_id = login_res["data"]["request_id"]
    
    totp_token = pyotp.TOTP(totp_secret).now()
    session.post("https://kite.zerodha.com/api/twofa", data={
        "user_id": user_id, 
        "request_id": request_id, 
        "twofa_value": totp_token, 
        "twofa_type": "totp"
    })
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    redirect_res = session.get(login_url, allow_redirects=True)
    request_token = parse_qs(urlparse(redirect_res.url).query)["request_token"][0]
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print("✅ Zerodha Authentication Successful.")
except Exception as e:
    print("❌ Error logging in:", e)
    sys.exit(1)

# Market Status Verification
current_time = datetime.datetime.now().time()
settlement_time = datetime.time(16, 0)
if current_time < settlement_time:
    print("⚠️ WARNING: It is before 4:00 PM IST. Post-CAS settlement might not be final.")

parquet_file = "nse_6yr_historical.parquet"
if not os.path.exists(parquet_file):
    print("❌ Parquet database not found.")
    sys.exit(1)

df_hist = pd.read_parquet(parquet_file)
unique_symbols = df_hist['Symbol'].unique()

kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunk_size = 200
chunks = [kite_symbols[i:i + chunk_size] for i in range(0, len(kite_symbols), chunk_size)]

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
new_rows = []

print(f"Fetching EOD snapshots for {len(kite_symbols)} stocks...")
for chunk in chunks:
    for attempt in range(5):
        try:
            res = kite.quote(chunk)
            if res:
                for symbol, data in res.items():
                    clean_symbol = symbol.replace("NSE:", "")
                    new_rows.append({
                        "Date": today_str,
                        "Open": data['ohlc']['open'],
                        "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'],
                        "Close": data['last_price'],
                        "Volume": data['volume'],
                        "Symbol": clean_symbol
                    })
                break
        except Exception:
            time.sleep(2)
    time.sleep(0.4)

if new_rows:
    new_eod_df = pd.DataFrame(new_rows)
    new_eod_df['Date'] = pd.to_datetime(new_eod_df['Date']).dt.normalize()
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
    
    today_dt = pd.to_datetime(today_str).normalize()
    df_hist = df_hist[df_hist['Date'] != today_dt]
    
    combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
    combined.to_parquet(parquet_file, index=False)
    print(f"✅ Final EOD Database updated! ({len(new_rows)} stocks saved)")
    
    print("Triggering Math Engine (analyze_6yr_data.py)...")
    os.system("python analyze_6yr_data.py")
else:
    print("⚠️ No EOD data fetched.")
