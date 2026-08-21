import pandas as pd
import requests
import time
import datetime
import os
import sys
import pyotp
import re
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs

api_key, api_secret, user_id, password, totp_secret = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_API_SECRET"), os.environ.get("KITE_USER_ID"), os.environ.get("KITE_PASSWORD"), os.environ.get("KITE_TOTP")

try:
    kite = KiteConnect(api_key=api_key)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    login_res = session.post("https://kite.zerodha.com/api/login", data={"user_id": user_id, "password": password}).json()
    totp_token = pyotp.TOTP(totp_secret).now()
    twofa_res = session.post("https://kite.zerodha.com/api/twofa", data={"user_id": user_id, "request_id": login_res["data"]["request_id"], "twofa_value": totp_token}).json()
    
    def get_token(url): return parse_qs(urlparse(url).query).get("request_token", [None])[0]
    res = session.get(f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}", allow_redirects=True)
    if "connect/authorize" in res.url:
        form_data = {"action": "accept", "api_key": api_key}
        for tag in re.findall(r'<input[^>]*>', res.text, re.IGNORECASE):
            name_match = re.search(r'name=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if name_match: form_data[name_match.group(1)] = re.search(r'value=["\']([^"\']*)["\']', tag, re.IGNORECASE).group(1) if re.search(r'value=["\']([^"\']*)["\']', tag, re.IGNORECASE) else ""
        if "sess_id" in parse_qs(urlparse(res.url).query): form_data["sess_id"] = parse_qs(urlparse(res.url).query)["sess_id"][0]
        res = session.post("https://kite.zerodha.com/connect/authorize", data=form_data, allow_redirects=True)
    
    request_token = next((get_token(r.url) for r in res.history + [res] if get_token(r.url)), None)
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
except Exception as e: sys.exit(1)

instruments = kite.instruments("NSE")
df_scripts = pd.DataFrame(instruments)
# Strict Universe Filter: EQ only, Exclude BE/BZ
nse_stocks = df_scripts[(df_scripts['instrument_type'] == 'EQ') & (~df_scripts['tradingsymbol'].str.endswith(('-BE', '-BZ', '-SM', '-ST')))]
tokens_to_fetch = nse_stocks[['tradingsymbol', 'instrument_token']].to_dict('records')

yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
all_ohlc_data = []

for stock in tokens_to_fetch:
    stock_data = []
    for chunk in [("2018-04-01", "2022-03-31"), ("2022-04-01", yesterday_str)]:
        for _ in range(3):
            try:
                res = kite.historical_data(stock['instrument_token'], chunk[0], chunk[1], "day")
                if res: stock_data.extend(res)
                break
            except Exception: time.sleep(1)
        time.sleep(0.35)
    
    if stock_data:
        df_temp = pd.DataFrame(stock_data).rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df_temp['Symbol'] = stock['tradingsymbol']
        df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.date
        all_ohlc_data.append(df_temp)

if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    final_df = final_df.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")
    
    tmp_file = "nse_6yr_historical.tmp.parquet"
    final_df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, "nse_6yr_historical.parquet")
    os.system("python analyze_6yr_data.py")
