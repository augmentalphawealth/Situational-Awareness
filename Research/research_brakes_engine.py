import pandas as pd
import numpy as np

def run_brakes_research():
    print("🚦 RUNNING BRAKES ENGINE RESEARCH...")
    df = pd.read_csv('validation_trade_scores.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure Core Regime Score exists
    if 'Score_Leadership_50_30_20' not in df.columns:
        df['Score_Leadership_50_30_20'] = (0.50 * df['MidPct50EMA'] + 0.30 * df['SmallPct50EMA'] + 0.20 * df['LargePct50EMA'])
    
    # 1. Aggregate Daily Breadth
    daily = df.groupby('Date').agg({
        'Score_Leadership_50_30_20': 'first',
        'FollowThroughRate': 'first',
        'MidPct20EMA': 'first',
        'MidPct50EMA': 'first',
        'Net52WHighLow': 'first',
        'Rolling3DDown4': 'first'
    }).sort_index().reset_index()

    # 2. Define Independent Brake Signals (Conditioned on High Regime)
    high_regime = daily['Score_Leadership_50_30_20'] >= 65
    
    daily['Sig_FTR_Collapse'] = high_regime & (daily['FollowThroughRate'] <= 40)
    daily['Sig_Mom_Diverge'] = high_regime & ((daily['MidPct20EMA'] - daily['MidPct50EMA']) < -10)
    daily['Sig_Narrowing'] = high_regime & (daily['Net52WHighLow'] < 5)
    daily['Sig_DownThrust'] = high_regime & (daily['Rolling3DDown4'] >= 180)

    # 3. Require at least TWO independent signals to trigger the "Brake" (Claude's Rule)
    daily['Brake_Signal_Count'] = daily[['Sig_FTR_Collapse', 'Sig_Mom_Diverge', 'Sig_Narrowing', 'Sig_DownThrust']].sum(axis=1)
    daily['Active_Brakes_Today'] = daily['Brake_Signal_Count'] >= 2

    # 4. SHIFT FORWARD 1 DAY to prevent look-ahead bias (Trade taken day AFTER signal)
    daily['Active_Brakes_NextDay'] = daily['Active_Brakes_Today'].shift(1).fillna(False)

    # 5. Merge back to Trade Data
    df = pd.merge(df, daily[['Date', 'Active_Brakes_NextDay']], on='Date', how='left')

    # 6. Evaluate
    strong_env = df[df['Score_Leadership_50_30_20'] >= 65]
    brakes_on = strong_env[strong_env['Active_Brakes_NextDay'] == True]
    brakes_off = strong_env[strong_env['Active_Brakes_NextDay'] == False]

    results = pd.DataFrame({
        'State': ['Regime >= 65 (No Brakes)', 'Regime >= 65 (BRAKES ACTIVE)'],
        'Trades': [len(brakes_off), len(brakes_on)],
        'Win_15Pct': [brakes_off['Success15'].mean() * 100, brakes_on['Success15'].mean() * 100],
        'Stop_7Pct': [brakes_off['Stop7'].mean() * 100, brakes_on['Stop7'].mean() * 100],
        'Avg_Rule_Return': [brakes_off['RuleBasedReturn'].mean(), brakes_on['RuleBasedReturn'].mean()]
    })

    print(results.to_string(index=False))
    results.to_csv('research_brakes_summary.csv', index=False)
    print("Saved to research_brakes_summary.csv\n")

if __name__ == "__main__":
    run_brakes_research()
