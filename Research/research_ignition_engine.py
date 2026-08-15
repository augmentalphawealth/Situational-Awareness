import pandas as pd
import numpy as np

def run_ignition_research():
    print("🚀 RUNNING IGNITION ENGINE RESEARCH...")
    df = pd.read_csv('validation_trade_scores.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure Core Regime Score exists
    if 'Score_Leadership_50_30_20' not in df.columns:
        df['Score_Leadership_50_30_20'] = (0.50 * df['MidPct50EMA'] + 0.30 * df['SmallPct50EMA'] + 0.20 * df['LargePct50EMA'])
    
    daily = df.groupby('Date').agg({
        'Score_Leadership_50_30_20': 'first',
        'Rolling3DUp4': 'first',
        'MidPct20EMA': 'first',
        'MidPct50EMA': 'first',
        'Net52WHighLow': 'first'
    }).sort_index().reset_index()

    # Calculate 5-day rate of change for 20EMA and New Highs/Lows
    daily['MidPct20EMA_5d_Change'] = daily['MidPct20EMA'].diff(5)
    daily['Net52WHighLow_5d_Change'] = daily['Net52WHighLow'].diff(5)

    # 1. Define Independent Ignition Signals (Conditioned on Weak Regime)
    low_regime = daily['Score_Leadership_50_30_20'] <= 40
    
    daily['Sig_Thrust'] = low_regime & (daily['Rolling3DUp4'] >= 600)
    daily['Sig_Mom_Surge'] = low_regime & (daily['MidPct20EMA_5d_Change'] >= 15) # Fast 20EMA recovery
    daily['Sig_Breadth_Repair'] = low_regime & (daily['MidPct20EMA'] > (daily['MidPct50EMA'] + 5))
    daily['Sig_Lows_Drying'] = low_regime & (daily['Net52WHighLow_5d_Change'] >= 20)

    # 2. Require at least TWO independent signals (Claude's Rule)
    daily['Ignition_Signal_Count'] = daily[['Sig_Thrust', 'Sig_Mom_Surge', 'Sig_Breadth_Repair', 'Sig_Lows_Drying']].sum(axis=1)
    daily['Active_Ignition_Today'] = daily['Ignition_Signal_Count'] >= 2

    # 3. SHIFT FORWARD 1 DAY to prevent look-ahead bias
    daily['Active_Ignition_NextDay'] = daily['Active_Ignition_Today'].shift(1).fillna(False)

    # 4. Merge back to Trade Data
    df = pd.merge(df, daily[['Date', 'Active_Ignition_NextDay']], on='Date', how='left')

    # 5. Evaluate
    weak_env = df[df['Score_Leadership_50_30_20'] <= 40]
    ignition_on = weak_env[weak_env['Active_Ignition_NextDay'] == True]
    ignition_off = weak_env[weak_env['Active_Ignition_NextDay'] == False]

    results = pd.DataFrame({
        'State': ['Regime <= 40 (No Ignition)', 'Regime <= 40 (IGNITION ACTIVE)'],
        'Trades': [len(ignition_off), len(ignition_on)],
        'Win_15Pct': [ignition_off['Success15'].mean() * 100, ignition_on['Success15'].mean() * 100],
        'Stop_7Pct': [ignition_off['Stop7'].mean() * 100, ignition_on['Stop7'].mean() * 100],
        'Avg_Rule_Return': [ignition_off['RuleBasedReturn'].mean(), ignition_on['RuleBasedReturn'].mean()]
    })

    print(results.to_string(index=False))
    results.to_csv('research_ignition_summary.csv', index=False)
    print("Saved to research_ignition_summary.csv\n")

if __name__ == "__main__":
    run_ignition_research()
