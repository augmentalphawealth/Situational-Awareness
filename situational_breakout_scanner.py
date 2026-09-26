"""
Situational Breakout Scanner / 2Y Backtest
-------------------------------------------
Daily NSE cash-equity model.

Information available at the close of day D is used to create a setup on D.
A breakout is confirmed only by a CLOSE above the prior consolidation high.
Entry is at the NEXT trading day's OPEN, so there is no same-bar/look-ahead use.

Setup:
1) 6-month momentum >= 30%.
2) Price > SMA50 > SMA200.
3) A meaningful prior thrust occurred inside the trailing 6 months.
4) Thrust had abnormal volume participation.
5) After the thrust, price pulled back but did not lose the trend structure.
6) Recent volume contracted versus its own 50D baseline.
7) Recent ATR% and range compressed.
8) Breakout must close above the immediately preceding consolidation range.

Ranking is deliberately continuous after the hard momentum/trend/liquidity gates.
The backtest records every qualifying breakout and forward 5/10/20/40-session returns,
plus MFE/MAE. It does not optimize on the test period.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("nse_6yr_historical.parquet")
OUT_DAILY = Path("situational_breakout_daily.csv")
OUT_TRADES = Path("situational_breakout_trades.csv")
OUT_SUMMARY = Path("situational_breakout_summary.csv")
OUT_SYMBOL = Path("situational_breakout_symbol_stats.csv")

LOOKBACK_6M = 126
MIN_MOMENTUM = 0.30
MIN_TURNOVER = 5e7                 # Rs 5 crore prior-20D avg
MIN_HISTORY = 210

# Thrust / pullback
THRUST_LOOKBACK = 126
MIN_THRUST = 0.30
THRUST_VOL_RATIO = 1.50
MIN_THRUST_2X_DAYS = 2
MAX_PULLBACK = 0.35
MIN_PULLBACK = 0.05

# Compression / consolidation
CONSOL_BARS = 10
MAX_CONSOL_RANGE = 0.10            # high/low <= 10%
MAX_ATR_PCT = 0.045
MAX_RECENT_VOL_RATIO = 0.80
BREAKOUT_VOL_MIN = 1.15

FORWARD_WINDOWS = [5, 10, 20, 40]


def prep(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date","Symbol","Open","High","Low","Close","Volume"])
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)].copy()
    df = df.sort_values(["Symbol","Date"]).drop_duplicates(["Symbol","Date"], keep="last")
    g = df.groupby("Symbol", group_keys=False)

    df["Turnover"] = df["Close"] * df["Volume"]
    df["ADV20_PRIOR"] = g["Turnover"].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())

    df["SMA50"] = g["Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    df["SMA200"] = g["Close"].transform(lambda x: x.rolling(200, min_periods=200).mean())
    df["RET_6M"] = g["Close"].transform(lambda x: x / x.shift(LOOKBACK_6M) - 1)

    prev = g["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.groupby(df["Symbol"]).transform(
        lambda x: x.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    )
    df["ATR_PCT"] = df["ATR14"] / df["Close"]
    df["VOL50"] = g["Volume"].transform(lambda x: x.shift(1).rolling(50, min_periods=50).mean())
    df["VOL_RATIO"] = df["Volume"] / df["VOL50"]

    # Previous-day-only breakout references. Current close can confirm the breakout.
    df["PRIOR_10_HIGH"] = g["High"].transform(lambda x: x.shift(1).rolling(CONSOL_BARS, min_periods=CONSOL_BARS).max())
    df["PRIOR_10_LOW"] = g["Low"].transform(lambda x: x.shift(1).rolling(CONSOL_BARS, min_periods=CONSOL_BARS).min())
    df["CONSOL_RANGE"] = df["PRIOR_10_HIGH"] / df["PRIOR_10_LOW"] - 1

    # Prior 10-day average volume is used to identify volume contraction.
    df["VOL20"] = g["Volume"].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    df["RECENT_VOL_RATIO"] = df["VOL20"] / df["VOL50"]

    return df


def features_for_row(g):
    g = g.sort_values("Date").reset_index(drop=True)
    n = len(g)
    if n < MIN_HISTORY:
        return None

    i = n - 1
    r = g.iloc[i]

    hard = (
        pd.notna(r["RET_6M"]) and r["RET_6M"] >= MIN_MOMENTUM and
        pd.notna(r["ADV20_PRIOR"]) and r["ADV20_PRIOR"] >= MIN_TURNOVER and
        pd.notna(r["SMA50"]) and pd.notna(r["SMA200"]) and
        r["Close"] > r["SMA50"] > r["SMA200"] and
        pd.notna(r["ATR_PCT"]) and pd.notna(r["PRIOR_10_HIGH"]) and
        pd.notna(r["RECENT_VOL_RATIO"])
    )
    if not hard:
        return None

    start = max(0, i - THRUST_LOOKBACK)
    hist = g.iloc[start:i].copy()
    if len(hist) < 60:
        return None

    # Identify the strongest low-to-subsequent-high thrust using only history before D.
    best = None
    lows = hist["Low"].to_numpy(float)
    highs = hist["High"].to_numpy(float)
    for a in range(0, len(hist) - 15):
        future = highs[a+1:]
        if len(future) == 0 or not np.isfinite(lows[a]) or lows[a] <= 0:
            continue
        b_rel = int(np.nanargmax(future))
        b = a + 1 + b_rel
        thrust = highs[b] / lows[a] - 1
        if thrust < MIN_THRUST:
            continue
        seg = hist.iloc[a:b+1]
        vr = seg["Volume"].to_numpy(float) / np.maximum(seg["Volume"].rolling(50, min_periods=1).mean().to_numpy(float), 1)
        days2x = int((vr >= 2.0).sum())
        cand = (thrust, a, b, days2x, float(np.nanmax(vr)))
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None or best[3] < MIN_THRUST_2X_DAYS:
        return None

    _, a, b, days2x, max_vr = best
    thrust_high = highs[b]

    # Pullback is measured from thrust high to current close, capped by recent low.
    post = g.iloc[start+b+1:i+1]
    if len(post) < 5:
        return None
    post_low = float(post["Low"].min())
    pullback = max(0.0, (thrust_high - post_low) / thrust_high)
    current_from_high = max(0.0, (thrust_high - r["Close"]) / thrust_high)

    if not (MIN_PULLBACK <= pullback <= MAX_PULLBACK):
        return None

    # Compression: recent 10-day range and ATR%, with volume dry-up.
    recent = g.iloc[i-CONSOL_BARS+1:i+1]
    if len(recent) < CONSOL_BARS:
        return None
    range10 = recent["High"].max() / recent["Low"].min() - 1
    atr_pct = float(r["ATR_PCT"])
    dry = float(r["RECENT_VOL_RATIO"])

    # A setup exists before the breakout. Breakout day is a separate event.
    setup_score = (
        min(max((r["RET_6M"] - .30) / .70, 0), 1) * 25 +
        min(max((best[0] - .30) / .70, 0), 1) * 20 +
        min(max((days2x - 2) / 6, 0), 1) * 10 +
        min(max((.80 - dry) / .50, 0), 1) * 15 +
        min(max((.10 - range10) / .07, 0), 1) * 15 +
        min(max((.045 - atr_pct) / .025, 0), 1) * 10 +
        min(max((.35 - pullback) / .30, 0), 1) * 5
    )

    breakout = (
        r["Close"] > r["PRIOR_10_HIGH"] and
        r["VOL_RATIO"] >= BREAKOUT_VOL_MIN
    )

    return {
        "Date": r["Date"],
        "Symbol": r["Symbol"],
        "Close": float(r["Close"]),
        "ADV20_RsCr": float(r["ADV20_PRIOR"] / 1e7),
        "Ret6M_pct": float(r["RET_6M"] * 100),
        "Thrust_pct": float(best[0] * 100),
        "ThrustDays2xVol": int(days2x),
        "MaxThrustVolRatio": float(best[4]),
        "Pullback_pct": float(pullback * 100),
        "CurrentFromThrustHigh_pct": float(current_from_high * 100),
        "Consol10Range_pct": float(range10 * 100),
        "ATR14_pct": float(atr_pct * 100),
        "RecentVolVs50_pct": float(dry * 100),
        "BreakoutLevel": float(r["PRIOR_10_HIGH"]),
        "DayVolRatio": float(r["VOL_RATIO"]),
        "Breakout": bool(breakout),
        "Score": float(setup_score),
    }


def build_daily(df):
    rows = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date")
        # Walk through every historical day. This is the key anti-look-ahead step.
        for i in range(MIN_HISTORY, len(g)):
            row = features_for_row(g.iloc[:i+1])
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_trades(df):
    # Recompute setup candidates day-by-day, then enter next trading day after a confirmed close breakout.
    daily = build_daily(df)
    if daily.empty:
        return daily, pd.DataFrame()

    all_trades = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        candidates = daily[daily["Symbol"] == sym]
        for _, c in candidates[candidates["Breakout"]].iterrows():
            idx = g.index[g["Date"] == c["Date"]]
            if len(idx) == 0:
                continue
            j = int(idx[0])
            if j + 1 >= len(g):
                continue
            entry = g.iloc[j+1]
            entry_price = float(entry["Open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            rec = c.to_dict()
            rec["EntryDate"] = entry["Date"]
            rec["EntryPrice"] = entry_price
            rec["GapFromBreakoutClose_pct"] = (entry_price / float(c["Close"]) - 1) * 100

            for w in FORWARD_WINDOWS:
                k = j + w
                if k < len(g):
                    f = g.iloc[k]
                    rec[f"Ret_{w}D_pct"] = (float(f["Close"]) / entry_price - 1) * 100
                    rec[f"MFE_{w}D_pct"] = (float(g.iloc[j+1:k+1]["High"].max()) / entry_price - 1) * 100
                    rec[f"MAE_{w}D_pct"] = (float(g.iloc[j+1:k+1]["Low"].min()) / entry_price - 1) * 100
                else:
                    rec[f"Ret_{w}D_pct"] = np.nan
                    rec[f"MFE_{w}D_pct"] = np.nan
                    rec[f"MAE_{w}D_pct"] = np.nan
            all_trades.append(rec)

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["EntryDate","Score"], ascending=[True,False]).reset_index(drop=True)
        # Avoid overlapping signals in the same symbol within 10 sessions for a cleaner event study.
        keep = []
        last_entry = {}
        for _, r in trades.iterrows():
            d = pd.Timestamp(r["EntryDate"])
            s = r["Symbol"]
            if s in last_entry and (d - last_entry[s]).days < 14:
                continue
            keep.append(r)
            last_entry[s] = d
        trades = pd.DataFrame(keep)
    return daily, trades


def summarize(trades):
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for w in FORWARD_WINDOWS:
        x = trades[f"Ret_{w}D_pct"].dropna()
        mfe = trades[f"MFE_{w}D_pct"].dropna()
        mae = trades[f"MAE_{w}D_pct"].dropna()
        if len(x) == 0:
            continue
        rows.append({
            "ForwardDays": w,
            "Trades": len(x),
            "WinRate_pct": (x > 0).mean() * 100,
            "MedianReturn_pct": x.median(),
            "MeanReturn_pct": x.mean(),
            "P25Return_pct": x.quantile(.25),
            "P75Return_pct": x.quantile(.75),
            "AvgMFE_pct": mfe.mean(),
            "MedianMFE_pct": mfe.median(),
            "AvgMAE_pct": mae.mean(),
            "MedianMAE_pct": mae.median(),
            "PctReturn_ge_5": (x >= 5).mean() * 100,
            "PctReturn_ge_10": (x >= 10).mean() * 100,
            "PctReturn_le_-5": (x <= -5).mean() * 100,
        })
    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df = prep(pd.read_parquet(INPUT))

    # Restrict event study to the latest two years, but keep full history as warm-up.
    max_date = df["Date"].max()
    test_start = max_date - pd.DateOffset(years=2)

    daily, trades = build_trades(df)
    daily = daily[daily["Date"] >= test_start].copy()
    trades = trades[trades["EntryDate"] >= test_start].copy()

    summary = summarize(trades)

    if not trades.empty:
        sym = trades.groupby("Symbol").agg(
            Trades=("Symbol","size"),
            WinRate_20D=("Ret_20D_pct", lambda x: (x.dropna() > 0).mean()*100),
            Median_20D=("Ret_20D_pct","median"),
            Mean_20D=("Ret_20D_pct","mean"),
            MedianMFE_20D=("MFE_20D_pct","median"),
            MedianMAE_20D=("MAE_20D_pct","median"),
        ).reset_index().sort_values(["Median_20D","Trades"], ascending=[False,False])
    else:
        sym = pd.DataFrame()

    daily.sort_values(["Date","Score"], ascending=[False,False]).to_csv(OUT_DAILY, index=False)
    trades.sort_values(["EntryDate","Score"], ascending=[False,False]).to_csv(OUT_TRADES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    sym.to_csv(OUT_SYMBOL, index=False)

    print(f"Data through: {max_date.date()}")
    print(f"Two-year test start: {test_start.date()}")
    print(f"Setup observations: {len(daily):,}")
    print(f"Breakout trades: {len(trades):,}")
    print("\nSUMMARY")
    print(summary.to_string(index=False) if not summary.empty else "No trades")
    if not trades.empty:
        print("\nRECENT TOP SIGNALS")
        print(trades.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
