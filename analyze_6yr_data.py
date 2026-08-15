import pandas as pd
import numpy as np
import datetime
import os

print("Loading 6-year Parquet database...")
parquet_file = "nse_6yr_historical.parquet"

try:
    df = pd.read_parquet(parquet_file)
except Exception as e:
    print(f"Error loading parquet file: {e}")
    exit()

print("Processing timestamps and cleaning data...")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# -------------------------------------------------------------
# STEP 0: SAFE GHOST DAY PURGE (VOLUME BASED)
# -------------------------------------------------------------
print("Purging holiday/weekend ghost data...")

# Hard purge of Saturdays (5) and Sundays (6)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df = df[df['DayOfWeek'] < 5].copy()
df.drop(columns=['DayOfWeek'], inplace=True)

daily_volume = df.groupby('Date')['Volume'].sum()
valid_dates = daily_volume[daily_volume > 0].index
df = df[df['Date'].isin(valid_dates)].reset_index(drop=True)

print("In-memory dataset cleaned. Master Parquet left untouched.")

# --- NEW CACHE SAVING STEP ---
print("Saving lightweight lookback cache for intraday syncs...")
# We only need enough history to calculate 200 EMAs and 252-day highs
cache_start_date = df['Date'].max() - pd.Timedelta(days=400)
cache_df = df[df['Date'] >= cache_start_date].reset_index(drop=True)
cache_df.to_parquet("trailing_cache.parquet", index=False)
print("✅ Lightweight cache saved to 'trailing_cache.parquet'.")

# -------------------------------------------------------------
# STEP 1: DYNAMIC TRAILING 45-DAY ROLLING TURNOVER CATEGORIZATION
# -------------------------------------------------------------
print("Calculating Daily Trailing 45-Day Rolling Turnover Ranks...")
df['Daily_Turnover'] = df['Close'] * df['Volume']

df['Turnover_45d_Avg'] = df.groupby('Symbol')['Daily_Turnover'].transform(
    lambda x: x.rolling(window=45, min_periods=10).mean()
)

df['Cap_Rank'] = df.groupby('Date')['Turnover_45d_Avg'].rank(ascending=False, method='min')

conditions = [
    (df['Cap_Rank'] <= 100),
    (df['Cap_Rank'] > 100) & (df['Cap_Rank'] <= 250),
    (df['Cap_Rank'] > 250) & (df['Cap_Rank'] <= 500)
]
choices = ['Top 100 Liq', 'Mid 150 Liq', 'Lower 250 Liq']
df['Liquidity_Category'] = np.select(conditions, choices, default='Micro Liq')

# -------------------------------------------------------------
# STEP 2: STRICT EMA & MOMENTUM CALCULATIONS
# -------------------------------------------------------------
print("Calculating EMAs and momentum indicators...")

df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=20).mean())
df['EMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=50).mean())
df['EMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=200).mean())

df['Daily_Pct'] = df.groupby('Symbol')['Close'].pct_change() * 100
df['Daily_Pct'] = df['Daily_Pct'].replace([np.inf, -np.inf], 0).fillna(0) 

df['Pct_1M'] = df.groupby('Symbol')['Close'].pct_change(periods=21) * 100

# -------------------------------------------------------------
# STEP 3: ADVANCE/DECLINE & TRUE RUPEE VOLUME BREADTH
# -------------------------------------------------------------
traded_today = df['Volume'] > 0
df['Gainer'] = (df['Daily_Pct'] > 0) & traded_today
df['Loser'] = (df['Daily_Pct'] < 0) & traded_today

df['Rolling_52W_High'] = df.groupby('Symbol')['High'].transform(lambda x: x.rolling(window=252, min_periods=200).max())
df['Rolling_52W_Low'] = df.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window=252, min_periods=200).min())

df['Above_20_EMA'] = df['Close'] > df['EMA_20']
df['Above_50_EMA'] = df['Close'] > df['EMA_50']
df['Above_200_EMA'] = df['Close'] > df['EMA_200']

df['Valid_20_EMA'] = df['EMA_20'].notna()
df['Valid_50_EMA'] = df['EMA_50'].notna()
df['Valid_200_EMA'] = df['EMA_200'].notna()

df['Up_4_Pct'] = (df['Daily_Pct'] >= 4.0) & traded_today
df['Down_4_Pct'] = (df['Daily_Pct'] <= -4.0) & traded_today
df['Up_25_1M'] = df['Pct_1M'] >= 25.0
df['Down_25_1M'] = df['Pct_1M'] <= -25.0

df['New_52W_High'] = df['Close'] >= df['Rolling_52W_High']
df['New_52W_Low'] = df['Close'] <= df['Rolling_52W_Low']

df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
df['ATR_14'] = df.groupby('Symbol')['TR'].transform(lambda x: x.rolling(14, min_periods=5).mean())
df['Vol_20D_Avg'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

df['20D_High'] = df.groupby('Symbol')['Close'].transform(lambda x: x == x.rolling(20, min_periods=20).max())
df['VCP_Tightness'] = (df['ATR_14'] / df['Close']) < 0.04
df['Volume_Surge'] = df['Volume'] > (df['Vol_20D_Avg'] * 1.5)

df['Is_Breakout'] = df['20D_High'] & df['Volume_Surge'] & df.groupby('Symbol')['VCP_Tightness'].shift(1)
df['Is_Breakout_3d_ago'] = df.groupby('Symbol')['Is_Breakout'].shift(3)
df['Close_3d_ago'] = df.groupby('Symbol')['Close'].shift(3)
df['Follow_Through_Win'] = (df['Is_Breakout_3d_ago'] == True) & (df['Close'] > df['Close_3d_ago'])

df['Up_Volume'] = np.where(df['Gainer'], df['Daily_Turnover'], 0)
df['Down_Volume'] = np.where(df['Loser'], df['Daily_Turnover'], 0)

# -------------------------------------------------------------
# STEP 4: CATEGORY AGGREGATIONS
# -------------------------------------------------------------
def get_liq_breadth(data, liq_name, prefix):
    liq_df = data[data['Liquidity_Category'] == liq_name]
    aggregated = liq_df.groupby('Date').agg(
        Valid_20=('Valid_20_EMA', 'sum'),
        Valid_50=('Valid_50_EMA', 'sum'),
        Valid_200=('Valid_200_EMA', 'sum'),
        Above_20=('Above_20_EMA', 'sum'),
        Above_50=('Above_50_EMA', 'sum'),
        Above_200=('Above_200_EMA', 'sum')
    ).reset_index()
    
    aggregated[f'{prefix}_Pct_20_EMA'] = (aggregated['Above_20'] / aggregated['Valid_20'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_50_EMA'] = (aggregated['Above_50'] / aggregated['Valid_50'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_200_EMA'] = (aggregated['Above_200'] / aggregated['Valid_200'].replace(0, np.nan)) * 100
    return aggregated[['Date', f'{prefix}_Pct_20_EMA', f'{prefix}_Pct_50_EMA', f'{prefix}_Pct_200_EMA']]

large_breadth = get_liq_breadth(df, 'Top 100 Liq', 'Large')
mid_breadth = get_liq_breadth(df, 'Mid 150 Liq', 'Mid')
small_breadth = get_liq_breadth(df, 'Lower 250 Liq', 'Small')
micro_breadth = get_liq_breadth(df, 'Micro Liq', 'Micro')

# -------------------------------------------------------------
# STEP 5: TOTAL UNIVERSE AGGREGATION
# -------------------------------------------------------------
overall_breadth = df.groupby('Date').agg(
    Total_Universe=('Symbol', 'count'),
    Valid_20=('Valid_20_EMA', 'sum'),
    Valid_50=('Valid_50_EMA', 'sum'),
    Valid_200=('Valid_200_EMA', 'sum'),
    Advances=('Gainer', 'sum'),
    Declines=('Loser', 'sum'),
    Above_20_EMA=('Above_20_EMA', 'sum'),
    Above_50_EMA=('Above_50_EMA', 'sum'),
    Above_200_EMA=('Above_200_EMA', 'sum'),
    Up_4_Count=('Up_4_Pct', 'sum'),
    Down_4_Count=('Down_4_Pct', 'sum'),
    Up_25_1M_Count=('Up_25_1M', 'sum'),
    Down_25_1M_Count=('Down_25_1M', 'sum'),
    New_52W_Highs=('New_52W_High', 'sum'),
    New_52W_Lows=('New_52W_Low', 'sum'),
    Total_Up_Volume=('Up_Volume', 'sum'),
    Total_Down_Volume=('Down_Volume', 'sum'),
    T3_Breakouts=('Is_Breakout_3d_ago', 'sum'),
    T3_Wins=('Follow_Through_Win', 'sum')
).reset_index()

overall_breadth['Pct_Above_20_EMA'] = (overall_breadth['Above_20_EMA'] / overall_breadth['Valid_20'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_50_EMA'] = (overall_breadth['Above_50_EMA'] / overall_breadth['Valid_50'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_200_EMA'] = (overall_breadth['Above_200_EMA'] / overall_breadth['Valid_200'].replace(0, np.nan)) * 100

overall_breadth['Rolling_3D_Up_4'] = overall_breadth['Up_4_Count'].rolling(window=3).sum()
overall_breadth['Rolling_3D_Down_4'] = overall_breadth['Down_4_Count'].rolling(window=3).sum()
overall_breadth['Slope_20_EMA'] = overall_breadth['Pct_Above_20_EMA'].diff(5)
overall_breadth['Slope_50_EMA'] = overall_breadth['Pct_Above_50_EMA'].diff(5)
overall_breadth['Slope_200_EMA'] = overall_breadth['Pct_Above_200_EMA'].diff(5)
overall_breadth['Net_52W_High_Low'] = overall_breadth['New_52W_Highs'] - overall_breadth['New_52W_Lows']

overall_breadth['Volume_Ratio'] = (overall_breadth['Total_Up_Volume'] / (overall_breadth['Total_Down_Volume'] + 1)).clip(upper=15.0)

final_summary = overall_breadth.merge(large_breadth, on='Date', how='left')
final_summary = final_summary.merge(mid_breadth, on='Date', how='left')
final_summary = final_summary.merge(small_breadth, on='Date', how='left')
final_summary = final_summary.merge(micro_breadth, on='Date', how='left')

final_summary = final_summary.drop(columns=['Valid_20', 'Valid_50', 'Valid_200'])

output_csv = "historical_breadth_regime_6yr.csv"
final_summary.to_csv(output_csv, index=False)
print(f"✅ Success! Analysis saved to '{output_csv}'.")

ist_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
current_ist_str = datetime.datetime.now(ist_offset).strftime('%d %b %Y, %I:%M %p IST')
with open("last_sync.txt", "w") as f:
    f.write(current_ist_str)
print(f"✅ Exact timestamp saved to last_sync.txt: {current_ist_str}")
