import pandas as pd
import requests
import time
import datetime
import os
import sys
from kiteconnect import KiteConnect

api_key = os.environ.get("KITE_API_KEY")
access_token = os.environ.get("KITE_ACCESS_TOKEN")

print("Connecting to Zerodha API for Historical Data Sync...")
try:
    if not api_key or not access_token:
        print("❌ KITE_API_KEY or KITE_ACCESS_TOKEN missing.")
        sys.exit(1)
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
except Exception as e:
    print("❌ Error connecting:", e)
    sys.exit(1)

instruments = kite.instruments("NSE")
df_scripts = pd.DataFrame(instruments)
nse_stocks = df_scripts[(df_scripts['instrument_type'] == 'EQ') & (~df_scripts['tradingsymbol'].str.endswith(('-BE', '-BZ', '-SM', '-ST')))]
tokens_to_fetch = nse_stocks[['tradingsymbol', 'instrument_token']].to_dict('records')

yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
all_ohlc_data = []

for stock in tokens_to_fetch:
    stock_data = []
    for chunk in [("2018-04-01", "2022-03-31"), ("2022-04-01", yesterday_str)]:
        for attempt in range(5):
            try:
                res = kite.historical_data(stock['instrument_token'], chunk[0], chunk[1], "day")
                if res: stock_data.extend(res)
                break
            except Exception as e:
                if "429" in str(e) or "403" in str(e):
                    time.sleep(2 ** attempt) # Exponential backoff for rate limits
                else:
                    time.sleep(1)
        # Guaranteed pacing to stay strictly under 3 req/sec
        time.sleep(0.4) 
    
    if stock_data:
        df_temp = pd.DataFrame(stock_data).rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df_temp['Symbol'] = stock['tradingsymbol']
        df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.date
        all_ohlc_data.append(df_temp)

if all_ohlc_data:
    final_df = pd.concat(all_ohlc_data, ignore_index=True)
    final_df = final_df.sort_values(by=['Symbol', 'Date']).drop_duplicates(subset=["Symbol", "Date"], keep="last")
    
    # 98% Completeness Guard
    unique_fetched = len(final_df['Symbol'].unique())
    if unique_fetched < 0.98 * len(tokens_to_fetch):
        print(f"❌ Completeness Guard Failed! Fetched {unique_fetched}/{len(tokens_to_fetch)} symbols. Aborting save.")
        with open("fetch_errors.log", "w") as f:
            f.write(f"[{datetime.datetime.now()}] Historical fetch failed 98% completeness threshold. Only got {unique_fetched} symbols.\n")
        sys.exit(1)

    tmp_file = "nse_6yr_historical.tmp.parquet"
    final_df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, "nse_6yr_historical.parquet")
    os.system("python analyze_6yr_data.py")
else:
    print("❌ No historical data fetched.")
