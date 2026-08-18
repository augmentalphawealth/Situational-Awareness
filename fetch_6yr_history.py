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

def rebuild_database():
    print("=========================================================")
    print("  ZERODHA KITE - FULL HISTORICAL REBUILD ENGINE")
    print("=========================================================")

    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    user_id = os.environ.get("KITE_USER_ID")
    password = os.environ.get("KITE_PASSWORD")
    totp_secret = os.environ.get("KITE_TOTP")

    print("Logging in to Zerodha...")
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

    print(f"Starting ADJUSTED historical data fetch for {total_stocks} stocks...")

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
            
        time.sleep(0.35) 

    if all_ohlc_data:
        final_df = pd.concat(all_ohlc_data, ignore_index=True)
        output_file = "nse_6yr_historical.parquet"
        final_df.to_parquet(output_file, index=False)
        print(f"✅ SUCCESS! ADJUSTED Database updated and saved to {output_file}.")
        
        # Trigger math calculations after rebuild
        print("Triggering Math Engine (analyze_6yr_data.py)...")
        os.system("python analyze_6yr_data.py")
    else:
        print("❌ Failed to fetch historical data.")
        sys.exit(1)

# Execute if run directly via GitHub Action
if __name__ == "__main__":
    rebuild_database()
    
