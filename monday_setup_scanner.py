import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = 'nse_6yr_historical.parquet'
OUT_ALL = 'monday_setup_candidates.csv'
OUT_READY = 'monday_ready.csv'
OUT_WATCH = 'monday_watchlist.csv'
MIN_TURNOVER_RS = 5_00_00_000
LOOKBACK_MONTHS = 6
MIN_THRUST_PCT = 20.0
MIN_2X_DAYS = 2
READY_VOL_RATIO_MAX = 0.70
READY_RANGE_MAX = 10.0


def dema(series, span):
    ema1 = series.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    return 2 * ema1 - ema2


def pct(x):
    return round(float(x), 2) if pd.notna(x) else np.nan


def process_symbol(g, latest_date):
    g = g.sort_values('Date').copy()
    if len(g) < 220:
        return None

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        g[col] = pd.to_numeric(g[col], errors='coerce')

    g = g.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    if len(g) < 220:
        return None

    g['TurnoverCalc'] = g['Close'] * g['Volume']
    g['ADV20'] = g['TurnoverCalc'].rolling(20).mean()
    g['VOL50'] = g['Volume'].rolling(50).mean()

    g['SMA20'] = g['Close'].rolling(20).mean()
    g['SMA50'] = g['Close'].rolling(50).mean()
    g['SMA200'] = g['Close'].rolling(200).mean()

    g['DEMA20'] = dema(g['Close'], 20)
    g['DEMA50'] = dema(g['Close'], 50)

    last = g.iloc[-1]

    if pd.isna(last['ADV20']) or last['ADV20'] < MIN_TURNOVER_RS:
        return None

    if pd.isna(last['SMA20']) or pd.isna(last['SMA50']) or pd.isna(last['SMA200']):
        return None

    # New strict trend filters
    if not (last['Close'] > last['SMA50'] and last['Close'] > last['SMA200']):
        return None

    if not (last['SMA50'] > last['SMA200']):
        return None

    if not (last['SMA20'] > last['SMA50']):
        return None

    start_date = latest_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    r = g[g['Date'] >= start_date].copy()
    if len(r) < 30:
        return None

    high_vals = r['High'].to_numpy(dtype=float)
    low_vals = r['Low'].to_numpy(dtype=float)

    best = None
    n = len(r)

    for i in range(0, n - 10):
        low_price = low_vals[i]
        if not np.isfinite(low_price) or low_price <= 0:
            continue

        future_high = np.nanmax(high_vals[i + 1:])
        if not np.isfinite(future_high):
            continue

        thrust_pct = (future_high / low_price - 1) * 100
        if thrust_pct < MIN_THRUST_PCT:
            continue

        j_rel = int(np.nanargmax(high_vals[i + 1:]))
        j = i + 1 + j_rel

        candidate = {
            'i': i,
            'j': j,
            'pivot_price': low_price,
            'pivot_date': r.iloc[i]['Date'],
            'high_price': high_vals[j],
            'high_date': r.iloc[j]['Date'],
            'thrust_pct': thrust_pct,
        }

        if best is None or candidate['thrust_pct'] > best['thrust_pct']:
            best = candidate

    if best is None:
        return None

    thrust_df = r.iloc[best['i']:best['j'] + 1].copy()
    thrust_df = thrust_df[thrust_df['VOL50'].notna()].copy()
    if len(thrust_df) < 3:
        return None

    thrust_df['VolRatio'] = thrust_df['Volume'] / thrust_df['VOL50']
    days_2x = int((thrust_df['VolRatio'] >= 2.0).sum())
    if days_2x < MIN_2X_DAYS:
        return None

    post = r[r['Date'] > best['high_date']].copy()
    if len(post) < 5:
        return None

    last10 = r.tail(10).copy()
    last5 = r.tail(5).copy()

    touch20 = bool(((last10['Low'] <= last10['DEMA20']) & (last10['High'] >= last10['DEMA20'])).any())
    touch50 = bool(((last10['Low'] <= last10['DEMA50']) & (last10['High'] >= last10['DEMA50'])).any())

    if not (touch20 or touch50):
        return None

    recent_vol_ratio = last5['Volume'].mean() / last['VOL50'] if pd.notna(last['VOL50']) and last['VOL50'] > 0 else np.nan
    last10_range_pct = (last10['High'].max() / last10['Low'].min() - 1) * 100 if last10['Low'].min() > 0 else np.nan
    post_high = post['High'].max()
    post_low = post['Low'].min()
    pullback_pct = ((post_high - post_low) / post_high) * 100 if pd.notna(post_high) and post_high > 0 else np.nan

    dist20 = abs((last['Close'] / last['DEMA20'] - 1) * 100) if pd.notna(last['DEMA20']) and last['DEMA20'] > 0 else np.nan
    dist50 = abs((last['Close'] / last['DEMA50'] - 1) * 100) if pd.notna(last['DEMA50']) and last['DEMA50'] > 0 else np.nan
    nearest_dema = np.nanmin([dist20, dist50]) if not (pd.isna(dist20) and pd.isna(dist50)) else np.nan

    dryup_score = max(0, (1.0 - recent_vol_ratio)) * 25 if pd.notna(recent_vol_ratio) else 0
    tightness_score = max(0, (12.0 - last10_range_pct)) * 1.5 if pd.notna(last10_range_pct) else 0
    thrust_score = best['thrust_pct'] * 0.5
    volume_score = min(float(thrust_df['VolRatio'].max()), 10.0) * 3 + days_2x * 2
    ma_score = max(0, (4 - nearest_dema)) * 3 if pd.notna(nearest_dema) else 0

    score = thrust_score + volume_score + dryup_score + tightness_score + ma_score

    bucket = 'Ready' if (
        pd.notna(recent_vol_ratio)
        and pd.notna(last10_range_pct)
        and recent_vol_ratio <= READY_VOL_RATIO_MAX
        and last10_range_pct <= READY_RANGE_MAX
    ) else 'Watchlist'

    return {
        'Symbol': str(g['Symbol'].iloc[0]),
        'LatestDate': latest_date.date().isoformat(),
        'ADV20_RsCr': pct(last['ADV20'] / 1e7),
        'PivotDate': best['pivot_date'].date().isoformat(),
        'PivotLow': pct(best['pivot_price']),
        'HighDate': best['high_date'].date().isoformat(),
        'ThrustHigh': pct(best['high_price']),
        'ThrustPct': pct(best['thrust_pct']),
        'Days2xVol': days_2x,
        'MaxVolRatio': pct(thrust_df['VolRatio'].max()),
        'AvgVolRatio_Thrust': pct(thrust_df['VolRatio'].mean()),
        'Recent5VolVs50': pct(recent_vol_ratio),
        'Last10RangePct': pct(last10_range_pct),
        'PullbackPct': pct(pullback_pct),
        'Touched20DEMA_10D': touch20,
        'Touched50DEMA_10D': touch50,
        'Dist20DEMA_Pct': pct(dist20),
        'Dist50DEMA_Pct': pct(dist50),
        'NearestDEMA_Pct': pct(nearest_dema),
        'CloseAbove50SMA': bool(last['Close'] > last['SMA50']),
        'CloseAbove200SMA': bool(last['Close'] > last['SMA200']),
        'SMA50Above200': bool(last['SMA50'] > last['SMA200']),
        'SMA20Above50': bool(last['SMA20'] > last['SMA50']),
        'Bucket': bucket,
        'Score': pct(score),
    }


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f'{INPUT_FILE} not found in repo root')

    if input_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    needed = {'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    latest_date = df['Date'].max()

    rows = []
    for _, g in df.groupby('Symbol', sort=False):
        row = process_symbol(g, latest_date)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        out = pd.DataFrame(columns=[
            'Symbol', 'LatestDate', 'ADV20_RsCr', 'PivotDate', 'PivotLow',
            'HighDate', 'ThrustHigh', 'ThrustPct', 'Days2xVol', 'MaxVolRatio',
            'AvgVolRatio_Thrust', 'Recent5VolVs50', 'Last10RangePct',
            'PullbackPct', 'Touched20DEMA_10D', 'Touched50DEMA_10D',
            'Dist20DEMA_Pct', 'Dist50DEMA_Pct', 'NearestDEMA_Pct',
            'CloseAbove50SMA', 'CloseAbove200SMA', 'SMA50Above200',
            'SMA20Above50', 'Bucket', 'Score'
        ])
    else:
        out = out.sort_values(
            ['Bucket', 'Score', 'ThrustPct', 'MaxVolRatio'],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

    ready = out[out['Bucket'] == 'Ready'].copy()
    watch = out[out['Bucket'] == 'Watchlist'].copy()

    out.to_csv(OUT_ALL, index=False)
    ready.to_csv(OUT_READY, index=False)
    watch.to_csv(OUT_WATCH, index=False)

    print('Latest date:', latest_date.date().isoformat())
    print('Total candidates:', len(out))
    print('Ready:', len(ready))
    print('Watchlist:', len(watch))

    if len(ready):
        print('\
Top Ready:')
        print(ready.head(20).to_string(index=False))

    if len(watch):
        print('\
Top Watchlist:')
        print(watch.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
