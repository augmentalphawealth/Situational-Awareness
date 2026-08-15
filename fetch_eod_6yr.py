import pandas as pd
import requests
import datetime
import pyotp
import os
import time
from SmartApi import SmartConnect
import logzero
import logging

# Mute the noisy Angel One logger
logzero.logger.setLevel(logging.FATAL)

api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

print("Logging in to Angel One for EOD Batch Sync...")
try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if session['status'] == False:
        print("Login Failed!")
        exit()
except Exception as e:
    print("Error logging in:", e)
    exit()

# -------------------------------------------------------------------
# MARKET STATUS & TIME CHECK
# -------------------------------------------------------------------
print("Verifying exchange activity and settlement time...")
current_time = datetime.datetime.now().time()
settlement_time = datetime.time(16, 0) # 4:00 PM IST (Post-CAS Settlement)

if current_time < settlement_time:
    print("⚠️ WARNING: It is before 4:00 PM. The Closing Auction Session (CAS) may not be fully settled.")
    print("Prices fetched right now might not be the final official EOD prices.")

parquet_file = "nse_6yr_historical.parquet"
if not os.path.exists(parquet_file):
    print("❌ Parquet database not found.")
    exit()

df_hist = pd.read_parquet(parquet_file)
unique_symbols = df_hist['Symbol'].unique()

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(url).json()
symbol_to_token = {item['symbol']: item['token'] for item in scrip_master if item['exch_seg'] == 'NSE' and item['symbol'].endswith('-EQ')}

all_tokens = [symbol_to_token[s] for s in unique_symbols if s in symbol_to_token]

# BATCH PROCESSING: Bypasses the blocked getCandleData firewall
chunk_size = 25
chunks = [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
new_rows = []

print(f"Fetching final EOD snapshots for {len(all_tokens)} stocks in batches...")
for chunk in chunks:
    for attempt in range(5):
        try:
            params = {"mode": "FULL", "exchangeTokens": {"NSE": chunk}}
            res = smartApi.getMarketData(params["mode"], params["exchangeTokens"])
            
            if res and isinstance(res, dict):
                if res.get('errorcode') == 'AB1021':
                    time.sleep(2) 
                    continue
                
                if res.get('status') and res.get('data'):
                    fetched_data = res['data'].get('fetched', [])
                    for item in fetched_data:
                        new_rows.append({
                            "Date": today_str,
                            "Open": item.get('open', 0),
                            "High": item.get('high', 0),
                            "Low": item.get('low', 0),
                            "Close": item.get('ltp', 0), # LTP acts as the settled Close post-4:00 PM
                            "Volume": item.get('tradeVolume', 0) or item.get('totBuyQuan', 0),
                            "Symbol": item.get('tradingSymbol', '')
                        })
                    break 
        except Exception:
            time.sleep(2)
            
    time.sleep(0.4)

if new_rows:
    new_eod_df = pd.DataFrame(new_rows)
    
    new_eod_df['Date'] = pd.to_datetime(new_eod_df['Date']).dt.normalize()
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
    
    # Remove any partial intraday data from today to prevent duplicates
    today_dt = pd.to_datetime(today_str).normalize()
    df_hist = df_hist[df_hist['Date'] != today_dt]
    
    # Append the final settled EOD data
    combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
    combined.to_parquet(parquet_file, index=False)
    print(f"✅ Final EOD Database updated successfully! ({len(new_rows)} stocks saved)")
    
    print("Triggering Math Engine to update Dashboard Analytics...")
    os.system("python analyze_6yr_data.py")
else:
    print("⚠️ No EOD data fetched. Market may be closed or tokens invalid.")
