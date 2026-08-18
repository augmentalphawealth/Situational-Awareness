import pandas as pd
import requests
import time
import datetime
import os
import sys
from SmartApi import SmartConnect
import pyotp

print("=========================================================")
print("  ANGEL ONE HISTORICAL FETCH - SECURE ENVIRONMENT ENGINE")
print("=========================================================")

api_key = os.environ.get("ANGEL_API_KEY")
client_code = os.environ.get("ANGEL_CLIENT_CODE")
login_pin = os.environ.get("ANGEL_PIN")
totp_secret = os.environ.get("ANGEL_TOTP")

if not all([api_key, client_code, login_pin, totp_secret]):
    print("❌ CRITICAL ERROR: Missing GitHub Secrets.")
    sys.exit(1)

try:
    smartApi = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = smartApi.generateSession(client_code, login_pin, totp)
    if session['status'] == False:
        print("❌ Login Failed! Verify your GitHub Secrets credentials.")
        sys.exit(1)
    print("✅ Angel One Authentication Successful.")
except Exception as e:
    print("❌ Error during authentication:", e)
    sys.exit(1)

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
scrip_master = requests.get(url).json()

df_scripts = pd.DataFrame(scrip_master)
nse_stocks = df_scripts[(df_scripts['exch_seg'] == 'NSE') & (df_scripts['symbol'].str.endswith('-EQ'))]
tokens_to_fetch = nse_stocks[['symbol', 'token']].to_dict('records')

end_date = datetime.datetime.now()
from_date_str = "2018-04-01 09:15"
to_date_str = end_date.strftime("%Y-%m-%d 15:30")

all_ohlc_data = []
total_stocks = len(tokens_to_fetch)
print(f"Starting historical data fetch for {total_stocks} stocks...")

for i, stock in enumerate(tokens_to_fetch):
    # MASSIVE RATE LIMIT FIX: 10 attempts, 15-second cooldown
    for attempt in range(10): 
        try:
            historicParam = {
                "exchange": "NSE",
                "symboltoken": stock['token'],
                "interval": "ONE_DAY",
                "fromdate": from_date_str, 
                "todate": to_date_str
            }
            hist_data = smartApi.getCandleData(historicParam)
            
            if hist_data and hist_data.get('status') == False:
                if hist_data.get('errorcode') == 'AB1021':
                    print(f"⚠️ Rate Limit Hit (AB1021) for token {stock['token']}. Cooling down for 15 seconds... (Attempt {attempt+1}/10)")
                    time.sleep(15) 
                    continue 
            
            if hist_data and hist_data.get('data'):
                df_temp = pd.DataFrame(hist_data['data'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df_temp['Symbol'] = stock['symbol']
                all_ohlc_data.append(df_temp)
            break 
            
        except Exception as e:
            time.sleep(2) 
            
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1} / {total_stocks} stocks fetched.")
        
    time.sleep(0.5) 

if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp']).dt.date
    final_df = final_df.rename(columns={'Timestamp': 'Date'})
    output_file = "nse_6yr_historical.parquet"
    final_df.to_parquet(output_file, index=False)
    print(f"✅ SUCCESS! Database updated from 01-Apr-2018 and saved to {output_file}.")
else:
    print("❌ Failed to fetch data.")
