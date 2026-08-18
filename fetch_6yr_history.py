import pandas as pd
import requests
import time
import datetime
import os
import sys
import pyotp
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs

print("=========================================================")
print("  ZERODHA KITE HISTORICAL FETCH - SECURE ENVIRONMENT ENGINE")
print("=========================================================")

# --- CREDENTIAL RETRIEVAL ---
api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")

missing_secrets = []
if not api_key: missing_secrets.append("KITE_API_KEY")
if not api_secret: missing_secrets.append("KITE_API_SECRET")
if not user_id: missing_secrets.append("KITE_USER_ID")
if not password: missing_secrets.append("KITE_PASSWORD")
if not totp_secret: missing_secrets.append("KITE_TOTP")

if missing_secrets:
    print(f"❌ CRITICAL ERROR: Missing GitHub Secrets: {', '.join(missing_secrets)}")
    sys.exit(1)

print("Logging in to Zerodha Kite via Automated TOTP...")
try:
    kite = KiteConnect(api_key=api_key)
    session = requests.Session()
    
    # Automated Headless Login Bypass
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
    print("❌ Error during authentication:", e)
    sys.exit(1)

print("Fetching Instruments Master from Zerodha...")
instruments = kite.instruments("NSE")
df_scripts = pd.DataFrame(instruments)
nse_stocks = df_scripts[df_scripts['instrument_type'] == 'EQ']
tokens_to_fetch = nse_stocks[['tradingsymbol', 'instrument_token']].to_dict('records')

from_date_str = "2018-04-01"
to_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

all_ohlc_data = []
total_stocks = len(tokens_to_fetch)

print(f"Starting ADJUSTED historical data fetch for {total_stocks} stocks (2018-04-01 to Present)...")

for i, stock in enumerate(tokens_to_fetch):
    for attempt in range(3):
        try:
            hist_data = kite.historical_data(
                instrument_token=stock['instrument_token'],
                from_date=from_date_str,
                to_date=to_date_str,
                interval="day"
            )
            
            if hist_data:
                df_temp = pd.DataFrame(hist_data)
                df_temp = df_temp.rename(columns={
                    'date': 'Date', 'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                })
                df_temp['Symbol'] = stock['tradingsymbol']
                if 'oi' in df_temp.columns:
                    df_temp = df_temp.drop(columns=['oi'])
                
                df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.date
                all_ohlc_data.append(df_temp)
            break 
        except Exception:
            time.sleep(1) 
            
    if (i + 1) % 150 == 0:
        print(f"Progress: {i + 1} / {total_stocks} stocks fetched.")
        
    time.sleep(0.35)  # Enforce Zerodha rate limits

if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    output_file = "nse_6yr_historical.parquet"
    final_df.to_parquet(output_file, index=False)
    print(f"✅ SUCCESS! ADJUSTED Database updated and saved to {output_file}.")
else:
    print("❌ Failed to fetch historical data.")
