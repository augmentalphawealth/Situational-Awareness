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

api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")

print("Logging in to Zerodha for EOD Batch Sync...")
try:
    kite = KiteConnect(api_key=api_key)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    login_res = session.post("https://kite.zerodha.com/api/login", data={"user_id": user_id, "password": password}).json()
    totp_token = pyotp.TOTP(totp_secret).now()
    twofa_res = session.post("https://kite.zerodha.com/api/twofa", data={"user_id": user_id, "request_id": login_res["data"]["request_id"], "twofa_value": totp_token}).json()
    
    def get_token(url): return parse_qs(urlparse(url).query).get("request_token", [None])[0]
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    res = session.get(login_url, allow_redirects=True)
    if "connect/authorize" in res.url:
        form_data = {}
        for tag in re.findall(r'<input[^>]*>', res.text, re.IGNORECASE):
            name_match = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if name_match: form_data[name_match.group(1)] = re.search(r'value=["\']([^"\']*)["\']', tag, re.IGNORECASE).group(1) if re.search(r'value=["\']([^"\']*)["\']', tag, re.IGNORECASE) else ""
        form_data.update({"action": "accept", "api_key": api_key})
        if "sess_id" in parse_qs(urlparse(res.url).query): form_data["sess_id"] = parse_qs(urlparse(res.url).query)["sess_id"][0]
        res = session.post("https://kite.zerodha.com/connect/authorize", data=form_data, allow_redirects=True)
    
    request_token = next((get_token(r.url) for r in res.history + [res] if get_token(r.url)), None)
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
except Exception as e:
    print("❌ Error logging in:", e)
    sys.exit(1)

parquet_file = "nse_6yr_historical.parquet"
tmp_parquet = "nse_6yr_historical.tmp.parquet"
if not os.path.exists(parquet_file): sys.exit(1)

df_hist = pd.read_parquet(parquet_file)
df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.normalize()
unique_symbols = df_hist['Symbol'].unique()

kite_symbols = [f"NSE:{sym}" for sym in unique_symbols]
chunks = [kite_symbols[i:i + 200] for i in range(0, len(kite_symbols), 200)]
today_dt = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d")).normalize()

new_rows = []
for chunk in chunks:
    for attempt in range(3):
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
        except Exception: time.sleep(1)
    time.sleep(1.1)  # Strictly respect 1 req/sec Quote API limit

if new_rows:
    new_eod_df = pd.DataFrame(new_rows)
    df_hist = df_hist[df_hist['Date'] != today_dt]
    combined = pd.concat([df_hist, new_eod_df], ignore_index=True)
    combined = combined.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")
    
    combined.to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, parquet_file)
    print("✅ EOD DB saved. Triggering Math...")
    os.system("python analyze_6yr_data.py")
