import pandas as pd
import numpy as np

def calculate_master_breadth(df):
    """
    Core engine for processing daily market data into institutional breadth metrics.
    Assumes df contains: Symbol, Date, Open, High, Low, Close, Volume
    """
    # Sort chronologically to prevent lookahead issues
    df = df.sort_values(by=['Symbol', 'Date'])

    # 1. Precise ATR & VCP Tightness (Wilder's Smoothing)
    df['Prev_Close'] = df.groupby('Symbol')['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Prev_Close']), 
                                     abs(df['Low'] - df['Prev_Close'])))
    # ewm with alpha=1/14 replicates standard industry ATR
    df['ATR_14'] = df.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    df['VCP_Tightness'] = (df['ATR_14'] / df['Close']) < 0.04

    # 2. Clean Volume Average (Shifted to prevent today's volume inflating the baseline)
    df['Vol_20D_Avg'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    df['Volume_Surge'] = df['Volume'] > (df['Vol_20D_Avg'] * 1.5)

    # 3. 20-Day High Breakout (Shifted baseline)
    df['Max_20D_Prior'] = df.groupby('Symbol')['High'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())
    df['Is_20D_High'] = df['Close'] > df['Max_20D_Prior']

    # 4. Breakout Trigger
    df['Is_Breakout'] = df['Is_20D_High'] & df['Volume_Surge'] & df.groupby('Symbol')['VCP_Tightness'].shift(1)

    # 5. Breakout Win-Rate (Pinning the win to the origin date, looking forward 3 days)
    df['Future_Close_3d'] = df.groupby('Symbol')['Close'].shift(-3)
    df['Follow_Through_Win'] = (df['Is_Breakout'] == True) & (df['Future_Close_3d'] > df['Close'])

    # 6. Moving Averages (Strict EMAs)
    df['EMA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['EMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())
    
    df['Above_20_EMA'] = df['Close'] > df['EMA_20']
    df['Above_50_EMA'] = df['Close'] > df['EMA_50']
    df['Above_200_EMA'] = df['Close'] > df['EMA_200']

    # 7. Advance / Decline / Volume Logic
    df['Is_Advance'] = df['Close'] > df['Prev_Close']
    df['Is_Decline'] = df['Close'] < df['Prev_Close']
    df['Up_Volume'] = np.where(df['Is_Advance'], df['Volume'], 0)
    df['Down_Volume'] = np.where(df['Is_Decline'], df['Volume'], 0)

    # 8. Aggregation Phase (Daily Roll-up)
    overall_breadth = df.groupby('Date').agg(
        Total_Universe=('Symbol', 'count'),
        Advances=('Is_Advance', 'sum'),
        Declines=('Is_Decline', 'sum'),
        Total_Up_Volume=('Up_Volume', 'sum'),
        Total_Down_Volume=('Down_Volume', 'sum'),
        Above_20_EMA=('Above_20_EMA', 'sum'),
        Above_50_EMA=('Above_50_EMA', 'sum'),
        Above_200_EMA=('Above_200_EMA', 'sum'),
        T3_Breakouts=('Is_Breakout', 'sum'),
        T3_Wins=('Follow_Through_Win', 'sum')
    ).reset_index()

    # Calculate Market Percentages
    overall_breadth['Pct_Above_20_EMA'] = (overall_breadth['Above_20_EMA'] / overall_breadth['Total_Universe']) * 100
    overall_breadth['Pct_Above_50_EMA'] = (overall_breadth['Above_50_EMA'] / overall_breadth['Total_Universe']) * 100
    overall_breadth['Pct_Above_200_EMA'] = (overall_breadth['Above_200_EMA'] / overall_breadth['Total_Universe']) * 100

    # 9. Institutional Tactical Indicators
    # McClellan Oscillator (MCO)
    overall_breadth['AD_Spread'] = overall_breadth['Advances'] - overall_breadth['Declines']
    overall_breadth['MCO'] = overall_breadth['AD_Spread'].ewm(span=19, adjust=False).mean() - overall_breadth['AD_Spread'].ewm(span=39, adjust=False).mean()

    # TRIN (Arms Index) - Includes Replace(0,1) to prevent ZeroDivisionError
    adv_dec_ratio = overall_breadth['Advances'] / overall_breadth['Declines'].replace(0, 1)
    vol_ratio = overall_breadth['Total_Up_Volume'] / overall_breadth['Total_Down_Volume'].replace(0, 1)
    overall_breadth['TRIN'] = adv_dec_ratio / vol_ratio

    return overall_breadth
