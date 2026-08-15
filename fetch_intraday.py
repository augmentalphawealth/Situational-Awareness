import pandas as pd
import requests
import datetime
import pyotp
import os
import time
from SmartApi import SmartConnect
import logzero
import logging

logzero.logger.setLevel(logging.FATAL)

api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

print("Logging in to Angel One for Intraday Breadth Snapshot...")
try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if session['status'] == False:
        print("❌ Login Failed!")
        exit()
except Exception as e:
    print("❌ Error logging in:", e)
    exit()

# Fetch Master Scrip List
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(url).json()
symbol_to_token = {item['symbol']: item['token'] for item in scrip_master if item['exch_seg'] == 'NSE' and item['symbol'].endswith('-EQ')}
all_tokens = list(symbol_to_token.values())

chunk_size = 50
chunks = [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]

# --- THE FIX: STRICT IST TIMEZONE BINDING ---
ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
now_ist = datetime.datetime.now(ist_offset)
today_str = now_ist.strftime("%Y-%m-%d")
time_now_str = now_ist.strftime("%H:%M")

new_rows = []

print(f"Fetching live intraday quotes for Advance/Decline calculation...")
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
                        # FULL OHLCV CAPTURE RESTORED. totBuyQuan fallback removed.
                        new_rows.append({
                            "Date": today_str,
                            "Symbol": item.get('tradingSymbol', ''),
                            "Open": item.get('open', 0),
                            "High": item.get('high', 0),
                            "Low": item.get('low', 0),
                            "Close": item.get('ltp', 0),
                            "Volume": item.get('tradeVolume', 0) 
                        })
                    break 
        except Exception:
            time.sleep(2)
    time.sleep(0.15)

if new_rows:
    live_df = pd.DataFrame(new_rows)
    
    # Calculate Live Market Breadth
    advances = live_df[live_df['Close'] > live_df['Open']]
    declines = live_df[live_df['Close'] < live_df['Open']]
    
    adv_cnt = len(advances)
    dec_cnt = len(declines)
    tot_cnt = adv_cnt + dec_cnt
    adv_pct_now = round((adv_cnt / tot_cnt * 100), 1) if tot_cnt > 0 else 50.0

    intraday_log_file = "live_intraday_breadth.csv"
    
    if os.path.exists(intraday_log_file):
        intra_df = pd.read_csv(intraday_log_file)
        if not intra_df.empty and str(intra_df['Date'].iloc[0]) != today_str:
            intra_df = pd.DataFrame(columns=["Time", "Adv_Pct", "Advances", "Declines", "Date"])
    else:
        intra_df = pd.DataFrame(columns=["Time", "Adv_Pct", "Advances", "Declines", "Date"])

    new_tick = pd.DataFrame([{"Time": time_now_str, "Adv_Pct": adv_pct_now, "Advances": adv_cnt, "Declines": dec_cnt, "Date": today_str}])
    intra_df = pd.concat([intra_df, new_tick], ignore_index=True).drop_duplicates(subset=['Time'], keep='last')
    intra_df.to_csv(intraday_log_file, index=False)
    
    print(f"✅ Real Intraday Breadth Snapshot Logged: {time_now_str} IST -> {adv_pct_now}% Advancing")
    print("✅ Master Parquet database left pristine and untouched.")
else:
    print("⚠️ No data fetched. API might be down.")
