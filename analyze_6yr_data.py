import pandas as pd
import numpy as np
import os
import datetime

print("=========================================================")
print("  EOD BREADTH & REGIME ANALYSIS ENGINE")
print("=========================================================")

parquet_file = "nse_6yr_historical.parquet"
if not os.path.exists(parquet_file):
    print("❌ Parquet database not found! Run fetch_6yr_history.py first.")
    exit()

df = pd.read_parquet(parquet_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

df['Daily_Turnover'] = df['Close'] * df['Volume']
df['Turnover_45d_Avg'] = df.groupby('Symbol')['Daily_Turnover'].transform(lambda x: x.shift(1).rolling(window=45, min_periods=10).mean())
df['Cap_Rank'] = df.groupby('Date')['Turnover_45d_Avg'].rank(ascending=False, method='min')

conditions = [(df['Cap_Rank'] <= 100), (df['Cap_Rank'] > 100) & (df['Cap_Rank'] <= 250), (df['Cap_Rank'] > 250) & (df['Cap_Rank'] <= 500)]
df['Liquidity_Category'] = np.select(conditions, ['Top 100 Liq', 'Mid 150 Liq', 'Lower 250 Liq'], default='Micro Liq')

df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=20).mean())
df['EMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=50).mean())
df['EMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False, min_periods=200).mean())

df['Daily_Pct'] = df.groupby('Symbol')['Close'].pct_change() * 100
df['Daily_Pct'] = df['Daily_Pct'].replace([np.inf, -np.inf], 0).fillna(0) 
df['Pct_1M'] = df.groupby('Symbol')['Close'].pct_change(periods=21) * 100

traded_today = df['Volume'] > 0
df['Gainer'] = (df['Daily_Pct'] > 0) & traded_today
df['Loser'] = (df['Daily_Pct'] < 0) & traded_today

df['Rolling_52W_High'] = df.groupby('Symbol')['High'].transform(lambda x: x.shift(1).rolling(window=252, min_periods=200).max())
df['Rolling_52W_Low'] = df.groupby('Symbol')['Low'].transform(lambda x: x.shift(1).rolling(window=252, min_periods=200).min())

df = df.dropna(subset=['EMA_200', 'Rolling_52W_High']).reset_index(drop=True)

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
df['New_52W_High'] = (df['Close'] >= df['Rolling_52W_High']) & traded_today
df['New_52W_Low'] = (df['Close'] <= df['Rolling_52W_Low']) & traded_today

# EOD Breakout (1.5x Volume Rule)
df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Prev_Close']), abs(df['Low'] - df['Prev_Close'])))
df['ATR_14'] = df.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
df['Vol_20D_Avg'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
df['Max_20D_Prior'] = df.groupby('Symbol')['High'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())

df['20D_High'] = df['Close'] > df['Max_20D_Prior']
df['VCP_Tightness'] = (df['ATR_14'] / df['Close']) < 0.04
df['Volume_Surge'] = df['Volume'] > (df['Vol_20D_Avg'] * 1.5)

prior_tight = df.groupby('Symbol')['VCP_Tightness'].shift(1).astype(float).fillna(0.0).astype(bool)
df['Is_Breakout'] = df['20D_High'] & df['Volume_Surge'] & prior_tight

df['Is_Breakout_3d_ago'] = df.groupby('Symbol')['Is_Breakout'].shift(3).astype(float).fillna(0.0).astype(bool)
df['Close_3d_ago'] = df.groupby('Symbol')['Close'].shift(3)
df['Follow_Through_Win'] = df['Is_Breakout_3d_ago'] & (df['Close'] > df['Close_3d_ago'])

df['Up_Volume'] = np.where(df['Gainer'], df['Daily_Turnover'], 0)
df['Down_Volume'] = np.where(df['Loser'], df['Daily_Turnover'], 0)

def get_liq_breadth(data, liq_name, prefix):
    liq_df = data[data['Liquidity_Category'] == liq_name]
    aggregated = liq_df.groupby('Date').agg(
        Valid_20=('Valid_20_EMA', 'sum'), Valid_50=('Valid_50_EMA', 'sum'), Valid_200=('Valid_200_EMA', 'sum'),
        Above_20=('Above_20_EMA', 'sum'), Above_50=('Above_50_EMA', 'sum'), Above_200=('Above_200_EMA', 'sum')
    ).reset_index()
    aggregated[f'{prefix}_Pct_20_EMA'] = (aggregated['Above_20'] / aggregated['Valid_20'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_50_EMA'] = (aggregated['Above_50'] / aggregated['Valid_50'].replace(0, np.nan)) * 100
    aggregated[f'{prefix}_Pct_200_EMA'] = (aggregated['Above_200'] / aggregated['Valid_200'].replace(0, np.nan)) * 100
    return aggregated[['Date', f'{prefix}_Pct_20_EMA', f'{prefix}_Pct_50_EMA', f'{prefix}_Pct_200_EMA']]

large_breadth = get_liq_breadth(df, 'Top 100 Liq', 'Large')
mid_breadth = get_liq_breadth(df, 'Mid 150 Liq', 'Mid')
small_breadth = get_liq_breadth(df, 'Lower 250 Liq', 'Small')
micro_breadth = get_liq_breadth(df, 'Micro Liq', 'Micro')

overall_breadth = df.groupby('Date').agg(
    Total_Universe=('Symbol', 'count'),
    Valid_20=('Valid_20_EMA', 'sum'), Valid_50=('Valid_50_EMA', 'sum'), Valid_200=('Valid_200_EMA', 'sum'),
    Advances=('Gainer', 'sum'), Declines=('Loser', 'sum'),
    Above_20_EMA=('Above_20_EMA', 'sum'), Above_50_EMA=('Above_50_EMA', 'sum'), Above_200_EMA=('Above_200_EMA', 'sum'),
    Up_4_Count=('Up_4_Pct', 'sum'), Down_4_Count=('Down_4_Pct', 'sum'),
    Up_25_1M_Count=('Up_25_1M', 'sum'), Down_25_1M_Count=('Down_25_1M', 'sum'),
    New_52W_Highs=('New_52W_High', 'sum'), New_52W_Lows=('New_52W_Low', 'sum'),
    Total_Up_Volume=('Up_Volume', 'sum'), Total_Down_Volume=('Down_Volume', 'sum'),
    T3_Breakouts=('Is_Breakout_3d_ago', 'sum'), T3_Wins=('Follow_Through_Win', 'sum')
).reset_index()

overall_breadth['Pct_Above_20_EMA'] = (overall_breadth['Above_20_EMA'] / overall_breadth['Valid_20'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_50_EMA'] = (overall_breadth['Above_50_EMA'] / overall_breadth['Valid_50'].replace(0, np.nan)) * 100
overall_breadth['Pct_Above_200_EMA'] = (overall_breadth['Above_200_EMA'] / overall_breadth['Valid_200'].replace(0, np.nan)) * 100

overall_breadth['Rolling_3D_Up_4'] = overall_breadth['Up_4_Count'].rolling(window=3).sum()
overall_breadth['Rolling_3D_Down_4'] = overall_breadth['Down_4_Count'].rolling(window=3).sum()
overall_breadth['Net_52W_High_Low'] = overall_breadth['New_52W_Highs'] - overall_breadth['New_52W_Lows']
overall_breadth['Volume_Ratio'] = overall_breadth['Total_Up_Volume'] / overall_breadth['Total_Down_Volume'].replace(0, np.nan)

overall_breadth['AD_Spread'] = overall_breadth['Advances'] - overall_breadth['Declines']
overall_breadth['MCO'] = overall_breadth['AD_Spread'].ewm(span=19, adjust=False).mean() - overall_breadth['AD_Spread'].ewm(span=39, adjust=False).mean()
overall_breadth['TRIN'] = (overall_breadth['Advances'] / overall_breadth['Declines'].replace(0, np.nan)) / (overall_breadth['Total_Up_Volume'] / overall_breadth['Total_Down_Volume'].replace(0, np.nan))

final_summary = overall_breadth.merge(large_breadth, on='Date', how='left').merge(mid_breadth, on='Date', how='left').merge(small_breadth, on='Date', how='left').merge(micro_breadth, on='Date', how='left')

# FULL 7-PART COMPOSITE SCORE CALCULATION
df_score = final_summary.copy()

p_blend = (0.65 * df_score['Pct_Above_20_EMA']) + (0.35 * df_score['Pct_Above_50_EMA'])
c1_breadth = (p_blend / 100) * 25

ft_rate = np.where(df_score['T3_Breakouts'] > 0, (df_score['T3_Wins'] / df_score['T3_Breakouts']) * 100, 0)
c2_breakout = np.where(ft_rate > 50, 25, 0)

net_4d = df_score['Rolling_3D_Up_4'] - df_score['Rolling_3D_Down_4']
net_1m = df_score['Up_25_1M_Count'] - df_score['Down_25_1M_Count']
rank_4d = net_4d.rolling(126, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
rank_1m = net_1m.rolling(126, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
c3_momentum = pd.Series((rank_4d * 10) + (rank_1m * 10)).fillna(0)

rank_vol = df_score['Volume_Ratio'].rolling(126, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
rank_hl = df_score['Net_52W_High_Low'].rolling(126, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
c4_vol_hl = pd.Series(np.where(df_score['Volume_Ratio'] > 1.0, rank_vol * 10, 0) + np.where(df_score['Net_52W_High_Low'] > 0, rank_hl * 10, 0)).fillna(0)

p200 = df_score['Pct_Above_200_EMA']
p200_slope = p200.diff(20) 
c5_lt = np.where((p200 > 50) & (p200_slope > 0), 10, np.where((p200 <= 50) & (p200_slope < 0), 0, 5))

hunting_ground = (df_score['Small_Pct_50_EMA'] + df_score['Micro_Pct_50_EMA']) / 2
gap = df_score['Large_Pct_50_EMA'] - hunting_ground
c6_penalty = np.where(gap >= 25, np.maximum(-15, (gap - 25) * -0.5), 0)

min_20d_p20 = df_score['Pct_Above_20_EMA'].rolling(20).min()
c7_bonus = np.where((min_20d_p20 <= 10) & (p_blend >= 50), 15, 0)

raw_score = pd.Series(c1_breadth) + pd.Series(c2_breakout) + c3_momentum + c4_vol_hl + pd.Series(c5_lt) + pd.Series(c6_penalty) + pd.Series(c7_bonus)
final_summary['Composite_Score'] = raw_score.clip(lower=0, upper=100).round().astype(int)

final_summary = final_summary.drop(columns=['Valid_20', 'Valid_50', 'Valid_200'])
output_csv = "historical_breadth_regime_6yr.csv"
final_summary.to_csv(output_csv, index=False)

cutoff_date = df['Date'].max() - pd.Timedelta(days=450)
df[df['Date'] >= cutoff_date].to_parquet("trailing_cache.parquet", index=False)
print("✅ Output Generated Successfully.")
