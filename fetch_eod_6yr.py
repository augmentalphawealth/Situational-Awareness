import pandas as pd
import requests
import datetime
import os
import sys
import time
import pyotp
import re
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
from fetch_6yr_history import rebuild_database

api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")

print("Logging in to Zerodha for EOD Batch Sync & Delta Check...")
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
        
    request_token = None
    
    def get_token(url):
        qs = parse_qs(urlparse(url).query)
        return qs.get("request_token", [None])[0]

    try:
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        res = session.get(login_url, allow_redirects=True)
        
        if "connect/authorize" in res.url:
            form_data = {}
            for tag in re.findall(r'<input[^>]*>', res.text, re.IGNORECASE):
                name_match = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                if name_match:
                    val_match = re.search(r'value=["\']([^"\']*)["\']', tag, re.IGNORECASE)
                    form_data[name_match.group(1)] = val_match.group(1) if val_match else ""
            
            if "action" not in form_data:
                form_data["action"] = "accept"
            
            query_params = parse_qs(urlparse(res.url).query)
            if "sess_id" in query_params and "sess_id" not in form_data:
                form_data["sess_id"] = query_params["sess_id"][0]
            if "api_key" not in form_data:
                form_data["api_key"] = api_key
            
            res = session.post("https://kite.zerodha.com/connect/authorize", data=form_data, allow_redirects=True)
        
        for resp in res.history + [res]:
            request_token = get_token(resp.url)
            if request_token: 
                break

    except requests.exceptions.RequestException as e:
        # Safely catch the 127.0.0.1 redirect crash and extract the token!
        if e.request and hasattr(e.request, 'url'):
            request_token = get_token(e.request.url)
            
    if not request_token:
        raise Exception("Failed to extract request token from redirect chain.")
        
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print("✅ Zerodha Authentication Successful.")
except Exception as e:
    print("❌ Error logging in:", e)
    sys.exit(1)

parquet_file = "nse_6yr_historical.parquet"
if not os.path.exists(parquet_file):
    print("❌ Parquet database not found. Initiating first-time rebuild...")
    rebuild_database()
    sys.exit(0)

print("Loading historical database for Delta Check...")
df_hist = pd.read_parquet(parquet_file)
unique_symbols = df_hist['Symbol'].unique()

# Map the last known Close price for every symbol in our database
last_closes = df_hist.groupby('Symbol').tail(1).set_index('Symbol')['Close'].to_dict()

kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunk_size = 200
chunks = [kite_symbols[i:i + chunk_size] for i in range(0, len(kite_symbols), chunk_size)]

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
new_rows = []
trigger_rebuild = False

print(f"Fetching EOD snapshots and verifying Deltas for {len(kite_symbols)} stocks...")
for chunk in chunks:
    if trigger_rebuild: 
        break
        
    chunk_success = False
    for attempt in range(5):
        try:
            res = kite.quote(chunk)
            if res:
                for symbol, data in res.items():
                    clean_symbol = symbol.replace("NSE:", "")
                    
                    # --- THE DELTA CHECK LOGIC ---
                    stored_prev_close = last_closes.get(clean_symbol)
                    actual_exchange_prev_close = data['ohlc']['close']
                    
                    if stored_prev_close and actual_exchange_prev_close > 0:
                        diff_pct = abs(stored_prev_close - actual_exchange_prev_close) / stored_prev_close
                        if diff_pct > 0.02: # If discrepancy > 2%, a corporate action occurred
                            print(f"\n🚨 CORPORATE ACTION DETECTED FOR {clean_symbol}!")
                            print(f"Stored Close: {stored_prev_close} | Exchange Adjusted Close: {actual_exchange_prev_close}")
                            trigger_rebuild = True
                            break
                    
                    # If no mismatch, prepare standard EOD row
                    new_rows.append({
                        "Date": today_str,
                        "Open": data['ohlc']['open'],
                        "High": data['ohlc']['high'],
                        "Low": data['ohlc']['low'],
                        "Close": data['last_price'],
                        "Volume": data['volume'],
                        "Symbol": clean_symbol
                    })
                chunk_success = True
                break
            else:
                print(f"⚠️ Zerodha returned empty data. Check symbol format! Sent: {chunk[:3]}")
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ Quote API Error (Attempt {attempt+1}/5): {e}")
            time.sleep(2)
            
    if not chunk_success:
        print(f"❌ Exhausted 5 attempts for chunk. Example symbols: {chunk[:3]}")
        
    time.sleep(0.4)

# --- ACTION HANDLING ---
if trigger_rebuild:
    print("\n⚠️ Database integrity compromised due to split/bonus. Wiping database and initiating full rebuild...\n")
    rebuild_database()
    print("✅ Automated Rebuild Complete. EOD Sync terminating.")
    sys.exit(0)

# If no corporate action detected, proceed with standard lightweight append
if new_rows:
    new_eod_df = pd.DataFrame(new_rows)
    new_eod_df['Date'] = pd.to_datetime(new_eod_df['Date']).dt.normalize()
    df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
    
    today_dt = pd.to_datetime(today_str).normalize()
    df_hist = df_hist[df_hist['Date'] != today_dt]
    
    combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
    combined.to_parquet(parquet_file, index=False)
    print(f"✅ Final EOD Database updated cleanly. No splits detected. ({len(new_rows)} stocks saved)")
    
    print("Triggering Math Engine (analyze_6yr_data.py)...")
    os.system("python analyze_6yr_data.py")
else:
    print("⚠️ No EOD data fetched.")
