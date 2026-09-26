"""
EMA Swing Trend Backtest
========================
Daily NSE cash-equity research engine using nse_6yr_historical.parquet.

Design principles
-----------------
1. All trend moving averages are DAILY exponential moving averages (EMA).
2. Signals use information available by the close of day D only.
3. Entry is the NEXT trading day's OPEN.
4. Initial structural stop must be <= the configured risk cap (4-6%).
5. Only one open trade per symbol per strategy combination.
6. Re-entry is allowed only after the previous trade is closed and a fresh signal appears.
7. Exit levels used intraday are based only on information known before that session.
8. Results are split into TRAIN and OOS (last two years) so the best historical fit is not
   automatically treated as the best deployable rule.

This engine intentionally compares several entry/compression variants with several exit
families. It does NOT choose a final live strategy automatically; the output files are
meant for robustness analysis after the workflow finishes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

INPUT = Path("nse_6yr_historical.parquet")
OUT_SETUPS = Path("swing_backtest_setups.csv")
OUT_TRADES = Path("swing_backtest_trades.csv")
OUT_COMPARE = Path("swing_backtest_strategy_comparison.csv")
OUT_YEARLY = Path("swing_backtest_yearly.csv")
OUT_SYMBOL = Path("swing_backtest_symbol_results.csv")
OUT_STABILITY = Path("swing_backtest_parameter_stability.csv")
OUT_RECOMMENDED = Path("swing_backtest_recommended_candidates.csv")
OUT_NOTES = Path("swing_backtest_notes.txt")

LOOKBACK_6M = 126
LOOKBACK_52W = 252
MIN_HISTORY = 260
MIN_MOMENTUM_6M = 0.30
MIN_ABOVE_52W_LOW = 0.50
MIN_TURNOVER_20D = 5e7
THRUST_LOOKBACK = 126
MIN_PRIOR_THRUST = 0.30
MIN_2X_VOLUME_DAYS = 2
MIN_PULLBACK = 0.05
MAX_PULLBACK = 0.30
OOS_YEARS = int(os.environ.get("OOS_YEARS", "2"))
ROUND_TRIP_COST_BPS = float(os.environ.get("ROUND_TRIP_COST_BPS", "20"))


@dataclass(frozen=True)
class EntryConfig:
    name: str
    bars: int
    max_range: float
    max_atr_pct: float
    max_vol_dry_ratio: float
    max_price_vol: float
    max_volume_cv: float
    breakout_vol_min: float
    stop_cap: float


ENTRY_CONFIGS: Tuple[EntryConfig, ...] = (
    EntryConfig("BALANCED_10D", 10, 0.08, 0.035, 0.70, 0.025, 0.80, 1.00, 0.06),
    EntryConfig("TIGHT_10D", 10, 0.06, 0.030, 0.60, 0.020, 0.70, 1.00, 0.05),
    EntryConfig("ULTRA_TIGHT_10D", 10, 0.05, 0.025, 0.50, 0.0175, 0.60, 1.10, 0.04),
    EntryConfig("LOW_RISK_10D", 10, 0.06, 0.030, 0.65, 0.020, 0.75, 1.00, 0.04),
    EntryConfig("FAST_5D", 5, 0.06, 0.035, 0.70, 0.030, 0.90, 1.00, 0.05),
    EntryConfig("BASE_15D", 15, 0.10, 0.035, 0.70, 0.025, 0.80, 1.00, 0.06),
    EntryConfig("BASE_20D", 20, 0.12, 0.040, 0.75, 0.030, 0.90, 1.00, 0.06),
    EntryConfig("NO_BREAKOUT_VOL_10D", 10, 0.08, 0.035, 0.70, 0.025, 0.80, 0.00, 0.06),
    EntryConfig("VOLUME_CONFIRM_10D", 10, 0.08, 0.035, 0.70, 0.025, 0.80, 1.30, 0.06),
)

EXIT_SPECS: Tuple[Tuple[str, str, float], ...] = (
    ("EMA10_CLOSE", "ema", 10.0),
    ("EMA20_CLOSE", "ema", 20.0),
    ("EMA50_CLOSE", "ema", 50.0),
    ("CHAND_2.0ATR", "chandelier", 2.0),
    ("CHAND_2.5ATR", "chandelier", 2.5),
    ("CHAND_3.0ATR", "chandelier", 3.0),
    ("CHAND_3.5ATR", "chandelier", 3.5),
    ("DONCHIAN_5D", "donchian", 5.0),
    ("DONCHIAN_10D", "donchian", 10.0),
    ("DONCHIAN_20D", "donchian", 20.0),
    ("SUPERTREND_2.5", "supertrend", 2.5),
    ("SUPERTREND_3.0", "supertrend", 3.0),
    ("SUPERTREND_3.5", "supertrend", 3.5),
)


def _clean_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Symbol", "Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical parquet missing required columns: {sorted(missing)}")
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    df = df[
        (df["Symbol"] != "") & (df["Open"] > 0) & (df["High"] > 0)
        & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] > 0)
        & (df["High"] >= df[["Open", "Close"]].max(axis=1))
        & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
    ].copy()
    return df.sort_values(["Symbol", "Date"]).drop_duplicates(["Symbol", "Date"], keep="last").reset_index(drop=True)


def _gt(df: pd.DataFrame, col: str, func):
    return df.groupby("Symbol", group_keys=False)[col].transform(func)


def _supertrend(g: pd.DataFrame, mult: float) -> np.ndarray:
    h, l, c = (g[x].to_numpy(float) for x in ["High", "Low", "Close"])
    atr = g["ATR10"].to_numpy(float)
    n = len(g)
    out = np.full(n, np.nan)
    if not np.isfinite(atr).any():
        return out
    hl2 = (h + l) / 2.0
    bu, bl = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    first = int(np.argmax(np.isfinite(atr)))
    fu[first], fl[first], out[first] = bu[first], bl[first], bl[first]
    for i in range(first + 1, n):
        if not np.isfinite(atr[i]):
            continue
        pfu = fu[i - 1] if np.isfinite(fu[i - 1]) else bu[i - 1]
        pfl = fl[i - 1] if np.isfinite(fl[i - 1]) else bl[i - 1]
        fu[i] = bu[i] if (bu[i] < pfu or c[i - 1] > pfu) else pfu
        fl[i] = bl[i] if (bl[i] > pfl or c[i - 1] < pfl) else pfl
        prev = out[i - 1]
        if not np.isfinite(prev):
            up = True
        elif math.isclose(prev, pfu, rel_tol=1e-10, abs_tol=1e-10):
            up = c[i] > fu[i]
        else:
            up = not (c[i] < fl[i])
        out[i] = fl[i] if up else fu[i]
    return out


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = _clean_input(raw)
    g = df.groupby("Symbol", group_keys=False)
    df["Bar"] = g.cumcount()
    df["HistoryDays"] = df["Bar"] + 1
    for span in (10, 20, 50, 200):
        df[f"EMA{span}"] = _gt(df, "Close", lambda x, s=span: x.ewm(span=s, adjust=False, min_periods=s).mean())
    df["Turnover"] = df["Close"] * df["Volume"]
    df["ADV20_PRIOR"] = _gt(df, "Turnover", lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    df["RET_6M"] = _gt(df, "Close", lambda x: x / x.shift(LOOKBACK_6M) - 1.0)
    df["LOW_52W_PRIOR"] = _gt(df, "Low", lambda x: x.shift(1).rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W).min())
    df["ABOVE_52W_LOW"] = df["Close"] / df["LOW_52W_PRIOR"] - 1.0

    prev = g["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR14"] = tr.groupby(df["Symbol"]).transform(lambda x: x.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    df["ATR10"] = tr.groupby(df["Symbol"]).transform(lambda x: x.ewm(alpha=1/10, adjust=False, min_periods=10).mean())
    df["ATR_PCT"] = df["ATR14"] / df["Close"]
    df["ATR_PCT_PRIOR5"] = _gt(df, "ATR_PCT", lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df["VOL50_PRIOR"] = _gt(df, "Volume", lambda x: x.shift(1).rolling(50, min_periods=50).mean())
    df["DAY_VOL_RATIO"] = df["Volume"] / df["VOL50_PRIOR"]
    df["RET1"] = g["Close"].pct_change()

    for w in (20, 40, 60, 90):
        df[f"RET_{w}D"] = _gt(df, "Close", lambda x, p=w: x / x.shift(p) - 1.0)
    df["THRUST_POINT"] = df[["RET_20D", "RET_40D", "RET_60D", "RET_90D"]].max(axis=1)
    df["HI_VOL_2X"] = (df["DAY_VOL_RATIO"] >= 2.0).astype(float)
    df["NextDate"] = g["Date"].shift(-1)
    df["NextOpen"] = g["Open"].shift(-1)

    for bars in sorted({c.bars for c in ENTRY_CONFIGS}):
        df[f"PRIOR_HIGH_{bars}"] = _gt(df, "High", lambda x, n=bars: x.shift(1).rolling(n, min_periods=n).max())
        df[f"PRIOR_LOW_{bars}"] = _gt(df, "Low", lambda x, n=bars: x.shift(1).rolling(n, min_periods=n).min())
        df[f"RANGE_{bars}"] = df[f"PRIOR_HIGH_{bars}"] / df[f"PRIOR_LOW_{bars}"] - 1.0
        df[f"VOL_AVG_{bars}"] = _gt(df, "Volume", lambda x, n=bars: x.shift(1).rolling(n, min_periods=n).mean())
        df[f"VOL_STD_{bars}"] = _gt(df, "Volume", lambda x, n=bars: x.shift(1).rolling(n, min_periods=n).std(ddof=0))
        df[f"VOL_DRY_{bars}"] = df[f"VOL_AVG_{bars}"] / df["VOL50_PRIOR"]
        df[f"VOL_CV_{bars}"] = df[f"VOL_STD_{bars}"] / df[f"VOL_AVG_{bars}"]
        df[f"PRICE_VOL_{bars}"] = _gt(df, "RET1", lambda x, n=bars: x.shift(1).rolling(n, min_periods=n).std(ddof=0))
        df[f"PRIOR_THRUST_{bars}"] = _gt(df, "THRUST_POINT", lambda x, n=bars: x.shift(n).rolling(THRUST_LOOKBACK, min_periods=40).max())
        df[f"PRIOR_2XVOL_DAYS_{bars}"] = _gt(df, "HI_VOL_2X", lambda x, n=bars: x.shift(n).rolling(THRUST_LOOKBACK, min_periods=40).sum())
        df[f"PREBASE_HIGH_{bars}"] = _gt(df, "High", lambda x, n=bars: x.shift(n).rolling(THRUST_LOOKBACK, min_periods=40).max())
        df[f"PULLBACK_{bars}"] = (df[f"PREBASE_HIGH_{bars}"] - df[f"PRIOR_LOW_{bars}"]) / df[f"PREBASE_HIGH_{bars}"]
        df[f"DONCH_TRAIL_{bars}"] = _gt(df, "Low", lambda x, n=bars: x.rolling(n, min_periods=n).min())

    for mult in (2.5, 3.0, 3.5):
        pieces = [pd.Series(_supertrend(sg, mult), index=sg.index) for _, sg in df.groupby("Symbol", sort=False)]
        df[f"ST_{mult:.1f}"] = pd.concat(pieces).sort_index() if pieces else np.nan
    return df


def fixed_selection_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["HistoryDays"] >= MIN_HISTORY)
        & (df["ADV20_PRIOR"] >= MIN_TURNOVER_20D)
        & (df["Close"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["EMA20"] > df["EMA50"])
        & (df["EMA50"] > df["EMA200"])
        & (df["RET_6M"] >= MIN_MOMENTUM_6M)
        & (df["ABOVE_52W_LOW"] >= MIN_ABOVE_52W_LOW)
        & df["NextOpen"].notna()
    )


def build_setups(df: pd.DataFrame) -> pd.DataFrame:
    fixed = fixed_selection_mask(df)
    all_setups: List[pd.DataFrame] = []
    for cfg in ENTRY_CONFIGS:
        n = cfg.bars
        stop = df[f"PRIOR_LOW_{n}"]
        entry = df["NextOpen"]
        risk = (entry - stop) / entry
        mask = (
            fixed
            & (df[f"PRIOR_THRUST_{n}"] >= MIN_PRIOR_THRUST)
            & (df[f"PRIOR_2XVOL_DAYS_{n}"] >= MIN_2X_VOLUME_DAYS)
            & (df[f"PULLBACK_{n}"] >= MIN_PULLBACK) & (df[f"PULLBACK_{n}"] <= MAX_PULLBACK)
            & (df[f"RANGE_{n}"] <= cfg.max_range)
            & (df["ATR_PCT_PRIOR5"] <= cfg.max_atr_pct)
            & (df[f"VOL_DRY_{n}"] <= cfg.max_vol_dry_ratio)
            & (df[f"PRICE_VOL_{n}"] <= cfg.max_price_vol)
            & (df[f"VOL_CV_{n}"] <= cfg.max_volume_cv)
            & (df["Close"] > df[f"PRIOR_HIGH_{n}"])
            & (df["DAY_VOL_RATIO"] >= cfg.breakout_vol_min)
            & (entry > stop) & (risk > 0) & (risk <= cfg.stop_cap)
        )
        cols = ["Symbol", "Date", "Bar", "Close", "NextDate", "NextOpen", "EMA20", "EMA50", "EMA200", "RET_6M", "ABOVE_52W_LOW", "ADV20_PRIOR", "ATR_PCT_PRIOR5", "DAY_VOL_RATIO"]
        s = df.loc[mask, cols].copy()
        if s.empty:
            continue
        s["EntryConfig"], s["ConsolBars"] = cfg.name, n
        s["BreakoutLevel"] = df.loc[mask, f"PRIOR_HIGH_{n}"].to_numpy()
        s["StructuralStop"] = stop.loc[mask].to_numpy()
        s["InitialRisk_pct"] = risk.loc[mask].to_numpy() * 100
        s["ConsolRange_pct"] = df.loc[mask, f"RANGE_{n}"].to_numpy() * 100
        s["VolDryRatio"] = df.loc[mask, f"VOL_DRY_{n}"].to_numpy()
        s["PriceVol_pct"] = df.loc[mask, f"PRICE_VOL_{n}"].to_numpy() * 100
        s["VolumeCV"] = df.loc[mask, f"VOL_CV_{n}"].to_numpy()
        s["PriorThrust_pct"] = df.loc[mask, f"PRIOR_THRUST_{n}"].to_numpy() * 100
        s["Prior2xVolDays"] = df.loc[mask, f"PRIOR_2XVOL_DAYS_{n}"].to_numpy()
        s["Pullback_pct"] = df.loc[mask, f"PULLBACK_{n}"].to_numpy() * 100
        s["SetupScore"] = (
            (s["RET_6M"].clip(.30, 1.50) - .30) * 20
            + (s["PriorThrust_pct"].clip(30, 150) - 30) * .15
            + (1 - s["VolDryRatio"].clip(0, 1)) * 15
            + (cfg.max_range * 100 - s["ConsolRange_pct"]).clip(lower=0) * 1.5
            + (cfg.stop_cap * 100 - s["InitialRisk_pct"]).clip(lower=0) * 2
        )
        all_setups.append(s)
    if not all_setups:
        return pd.DataFrame()
    out = pd.concat(all_setups, ignore_index=True).rename(columns={"Date": "SignalDate", "NextDate": "EntryDate", "NextOpen": "EntryPrice"})
    out["EntryBar"] = out["Bar"].astype(int) + 1
    return out.sort_values(["EntryConfig", "Symbol", "EntryDate", "SetupScore"], ascending=[True, True, True, False])


def _stop_hit(op: float, lo: float, stop: float):
    if not np.isfinite(stop) or stop <= 0:
        return False, np.nan, ""
    if op <= stop:
        return True, float(op), "GAP_THROUGH_STOP"
    if lo <= stop:
        return True, float(stop), "STOP"
    return False, np.nan, ""


def simulate_one_trade(sg, entry_bar, entry_price, initial_stop, exit_name, exit_kind, exit_param):
    local = sg
    last_bar = int(local.index.max())
    if entry_bar not in local.index:
        return {}
    highest, trail, mfe, mae = -np.inf, float(initial_stop), -np.inf, np.inf
    exit_bar, exit_price, exit_reason = last_bar, float(local.loc[last_bar, "Close"]), "END_OF_DATA"
    for bar in range(entry_bar, last_bar + 1):
        if bar not in local.index:
            continue
        r = local.loc[bar]
        op, hi, lo, cl = (float(r[x]) for x in ["Open", "High", "Low", "Close"])
        hit, px, reason = _stop_hit(op, lo, trail)
        mfe, mae = max(mfe, hi / entry_price - 1), min(mae, lo / entry_price - 1)
        if hit:
            exit_bar, exit_price = bar, px
            exit_reason = reason if trail <= initial_stop + 1e-12 else f"{exit_name}_{reason}"
            break
        highest = max(highest, hi)
        if exit_kind == "ema":
            ema = float(r[f"EMA{int(exit_param)}"])
            if np.isfinite(ema) and cl < ema:
                nb = bar + 1
                exit_bar, exit_price, exit_reason = (nb, float(local.loc[nb, "Open"]), exit_name) if nb in local.index else (bar, cl, f"{exit_name}_EOD")
                break
        elif exit_kind == "supertrend":
            st = float(r[f"ST_{exit_param:.1f}"])
            if np.isfinite(st) and cl < st:
                nb = bar + 1
                exit_bar, exit_price, exit_reason = (nb, float(local.loc[nb, "Open"]), exit_name) if nb in local.index else (bar, cl, f"{exit_name}_EOD")
                break
        if exit_kind == "chandelier":
            atr = float(r["ATR14"])
            if np.isfinite(atr) and np.isfinite(highest):
                trail = max(trail, highest - exit_param * atr)
        elif exit_kind == "donchian":
            v = float(r[f"DONCH_TRAIL_{int(exit_param)}"])
            if np.isfinite(v):
                trail = max(trail, v)
    gross = exit_price / entry_price - 1
    net = gross - ROUND_TRIP_COST_BPS / 10000
    er = local.loc[exit_bar]
    return {
        "ExitBar": int(exit_bar), "ExitDate": pd.Timestamp(er["Date"]), "ExitPrice": float(exit_price), "ExitReason": exit_reason,
        "GrossReturn_pct": gross * 100, "NetReturn_pct": net * 100,
        "MFE_pct": mfe * 100 if np.isfinite(mfe) else np.nan, "MAE_pct": mae * 100 if np.isfinite(mae) else np.nan,
        "ProfitGiveback_pct": max(0, mfe - gross) * 100 if np.isfinite(mfe) else np.nan,
        "HoldingSessions": int(exit_bar - entry_bar + 1),
    }


def run_backtests(df: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    frames = {sym: g.sort_values("Bar").set_index("Bar", drop=False) for sym, g in df.groupby("Symbol", sort=False)}
    trades: List[Dict[str, object]] = []
    for cfg, cs in setups.groupby("EntryConfig", sort=False):
        for exit_name, exit_kind, exit_param in EXIT_SPECS:
            for sym, ss in cs.groupby("Symbol", sort=False):
                sg = frames.get(sym)
                if sg is None:
                    continue
                last_exit = -1
                seen = set()
                for _, setup in ss.sort_values(["EntryBar", "SetupScore"], ascending=[True, False]).iterrows():
                    eb = int(setup["EntryBar"])
                    if eb in seen:
                        continue
                    seen.add(eb)
                    if (eb - 1) <= last_exit:
                        continue
                    result = simulate_one_trade(sg, eb, float(setup["EntryPrice"]), float(setup["StructuralStop"]), exit_name, exit_kind, exit_param)
                    if not result:
                        continue
                    rec = {
                        "EntryConfig": cfg, "ExitStrategy": exit_name, "Symbol": sym,
                        "SignalDate": setup["SignalDate"], "EntryDate": setup["EntryDate"], "EntryBar": eb,
                        "EntryPrice": float(setup["EntryPrice"]), "StructuralStop": float(setup["StructuralStop"]),
                        "InitialRisk_pct": float(setup["InitialRisk_pct"]), "SetupScore": float(setup["SetupScore"]),
                        "ConsolBars": int(setup["ConsolBars"]), "ConsolRange_pct": float(setup["ConsolRange_pct"]),
                        "VolDryRatio": float(setup["VolDryRatio"]), "PriceVol_pct": float(setup["PriceVol_pct"]),
                        "VolumeCV": float(setup["VolumeCV"]), "PriorThrust_pct": float(setup["PriorThrust_pct"]),
                        "Pullback_pct": float(setup["Pullback_pct"]), "Ret6M_pct": float(setup["RET_6M"] * 100),
                        "Above52WLow_pct": float(setup["ABOVE_52W_LOW"] * 100),
                    }
                    rec.update(result)
                    trades.append(rec)
                    last_exit = int(result["ExitBar"])
    if not trades:
        return pd.DataFrame()
    out = pd.DataFrame(trades)
    out["EntryDate"], out["ExitDate"] = pd.to_datetime(out["EntryDate"]), pd.to_datetime(out["ExitDate"])
    return out.sort_values(["EntryConfig", "ExitStrategy", "EntryDate", "Symbol"]).reset_index(drop=True)


def _pf(x: pd.Series) -> float:
    wins, losses = x[x > 0].sum(), -x[x < 0].sum()
    return np.inf if losses <= 0 and wins > 0 else (float(wins / losses) if losses > 0 else np.nan)


def summarize(t: pd.DataFrame, label: str) -> Dict[str, object]:
    if t.empty:
        return {"Period": label, "Trades": 0}
    r = t["NetReturn_pct"].dropna()
    return {
        "Period": label, "Trades": len(t), "Symbols": t["Symbol"].nunique(), "WinRate_pct": (r > 0).mean() * 100,
        "MeanNetReturn_pct": r.mean(), "MedianNetReturn_pct": r.median(),
        "GeoMeanPerTrade_pct": (np.exp(np.log1p(r / 100).mean()) - 1) * 100, "ProfitFactor": _pf(r),
        "AvgHoldingSessions": t["HoldingSessions"].mean(), "MedianHoldingSessions": t["HoldingSessions"].median(),
        "AvgInitialRisk_pct": t["InitialRisk_pct"].mean(), "AvgMFE_pct": t["MFE_pct"].mean(), "AvgMAE_pct": t["MAE_pct"].mean(),
        "AvgProfitGiveback_pct": t["ProfitGiveback_pct"].mean(), "PctTrades_ge_10": (r >= 10).mean() * 100,
        "PctTrades_ge_20": (r >= 20).mean() * 100, "PctTrades_le_minus5": (r <= -5).mean() * 100,
        "ReturnPerHoldingDay_bp": (r / t["HoldingSessions"].clip(lower=1)).mean() * 100,
    }


def build_symbol_results(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ec, ex, sym), t in trades.groupby(["EntryConfig", "ExitStrategy", "Symbol"]):
        t = t.sort_values("EntryDate")
        growth = np.prod(1 + t["NetReturn_pct"].to_numpy(float) / 100) - 1
        underlying = float(t.iloc[-1]["ExitPrice"]) / float(t.iloc[0]["EntryPrice"]) - 1
        rows.append({
            "EntryConfig": ec, "ExitStrategy": ex, "Symbol": sym, "Trades": len(t),
            "FirstEntryDate": t.iloc[0]["EntryDate"], "LastExitDate": t.iloc[-1]["ExitDate"],
            "StrategyCompoundedReturn_pct": growth * 100, "UnderlyingFirstEntryToLastExit_pct": underlying * 100,
            "TrendMonetisationRatio": growth / underlying if underlying > 0 else np.nan,
            "WinRate_pct": (t["NetReturn_pct"] > 0).mean() * 100, "AvgHoldingSessions": t["HoldingSessions"].mean(),
            "AvgGiveback_pct": t["ProfitGiveback_pct"].mean(),
        })
    return pd.DataFrame(rows)


def build_comparison(trades, symbol_results, max_date):
    if trades.empty:
        return pd.DataFrame()
    oos = max_date - pd.DateOffset(years=OOS_YEARS)
    rows = []
    for (ec, ex), t in trades.groupby(["EntryConfig", "ExitStrategy"]):
        for label, sub in (("ALL", t), ("TRAIN", t[t["EntryDate"] < oos]), ("OOS", t[t["EntryDate"] >= oos])):
            rec = {"EntryConfig": ec, "ExitStrategy": ex}
            rec.update(summarize(sub, label))
            sr = symbol_results[(symbol_results["EntryConfig"] == ec) & (symbol_results["ExitStrategy"] == ex)]
            if not sr.empty:
                rec["MedianSymbolStrategyReturn_pct"] = sr["StrategyCompoundedReturn_pct"].median()
                rec["MedianTrendMonetisationRatio"] = sr["TrendMonetisationRatio"].median()
                rec["PctSymbolsTrendMonetisation_gt_1"] = (sr["TrendMonetisationRatio"] > 1).mean() * 100
                rec["PctSymbolsTrendMonetisation_gt_1_5"] = (sr["TrendMonetisationRatio"] > 1.5).mean() * 100
            rows.append(rec)
    return pd.DataFrame(rows)


def build_yearly(trades):
    if trades.empty:
        return pd.DataFrame()
    x = trades.copy(); x["Year"] = x["EntryDate"].dt.year
    rows = []
    for (ec, ex, year), t in x.groupby(["EntryConfig", "ExitStrategy", "Year"]):
        rec = {"EntryConfig": ec, "ExitStrategy": ex, "Year": int(year)}; rec.update(summarize(t, str(year))); rows.append(rec)
    return pd.DataFrame(rows)


def build_stability(comparison):
    if comparison.empty:
        return pd.DataFrame()
    keep = ["EntryConfig", "ExitStrategy", "Trades", "WinRate_pct", "MeanNetReturn_pct", "MedianNetReturn_pct", "GeoMeanPerTrade_pct", "ProfitFactor", "AvgHoldingSessions", "AvgProfitGiveback_pct", "ReturnPerHoldingDay_bp", "MedianTrendMonetisationRatio"]
    train = comparison[comparison["Period"] == "TRAIN"][[c for c in keep if c in comparison.columns]].add_prefix("Train_")
    oos = comparison[comparison["Period"] == "OOS"][[c for c in keep if c in comparison.columns]].add_prefix("OOS_")
    train = train.rename(columns={"Train_EntryConfig": "EntryConfig", "Train_ExitStrategy": "ExitStrategy"})
    oos = oos.rename(columns={"OOS_EntryConfig": "EntryConfig", "OOS_ExitStrategy": "ExitStrategy"})
    s = train.merge(oos, on=["EntryConfig", "ExitStrategy"], how="outer")
    if "Train_MeanNetReturn_pct" in s and "OOS_MeanNetReturn_pct" in s:
        s["MeanReturn_OOS_minus_Train_pct"] = s["OOS_MeanNetReturn_pct"] - s["Train_MeanNetReturn_pct"]
    if "Train_GeoMeanPerTrade_pct" in s and "OOS_GeoMeanPerTrade_pct" in s:
        s["GeoMean_OOS_minus_Train_pct"] = s["OOS_GeoMeanPerTrade_pct"] - s["Train_GeoMeanPerTrade_pct"]
    return s


def build_recommended(stability):
    if stability.empty:
        return pd.DataFrame()
    req = ["Train_Trades", "OOS_Trades", "Train_GeoMeanPerTrade_pct", "OOS_GeoMeanPerTrade_pct"]
    if any(c not in stability.columns for c in req):
        return pd.DataFrame()
    e = stability[(stability["Train_Trades"] >= 30) & (stability["OOS_Trades"] >= 15)].copy()
    if e.empty:
        e = stability.copy()
    def col(name, default=0):
        return e[name].fillna(default) if name in e else pd.Series(default, index=e.index)
    e["RobustnessScore"] = (
        col("OOS_GeoMeanPerTrade_pct", -99) * 2
        + col("OOS_ReturnPerHoldingDay_bp") * .02
        + col("OOS_MedianTrendMonetisationRatio") * 1.5
        - col("OOS_AvgProfitGiveback_pct") * .10
        - col("GeoMean_OOS_minus_Train_pct").abs() * .5
    )
    return e.sort_values("RobustnessScore", ascending=False).head(25)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing historical database: {INPUT}")
    print("Loading historical NSE database...")
    raw = pd.read_parquet(INPUT)
    df = prepare_features(raw)
    max_date, min_date = pd.Timestamp(df["Date"].max()), pd.Timestamp(df["Date"].min())
    print(f"Rows={len(df):,}; symbols={df['Symbol'].nunique():,}; data={min_date.date()} to {max_date.date()}")
    setups = build_setups(df)
    print(f"Qualifying setup/entry observations: {len(setups):,}")
    if setups.empty:
        for p in [OUT_SETUPS, OUT_TRADES, OUT_COMPARE, OUT_YEARLY, OUT_SYMBOL, OUT_STABILITY, OUT_RECOMMENDED]:
            pd.DataFrame().to_csv(p, index=False)
        OUT_NOTES.write_text("No qualifying setups were found.\n", encoding="utf-8")
        return
    setups.to_csv(OUT_SETUPS, index=False)
    print(f"Running {len(ENTRY_CONFIGS)} entry variants x {len(EXIT_SPECS)} exit variants...")
    trades = run_backtests(df, setups)
    trades.to_csv(OUT_TRADES, index=False)
    symbol = build_symbol_results(trades)
    comp = build_comparison(trades, symbol, max_date)
    yearly = build_yearly(trades)
    stability = build_stability(comp)
    recommended = build_recommended(stability)
    symbol.to_csv(OUT_SYMBOL, index=False); comp.to_csv(OUT_COMPARE, index=False); yearly.to_csv(OUT_YEARLY, index=False)
    stability.to_csv(OUT_STABILITY, index=False); recommended.to_csv(OUT_RECOMMENDED, index=False)
    oos_start = max_date - pd.DateOffset(years=OOS_YEARS)
    OUT_NOTES.write_text(f"""EMA SWING TREND BACKTEST

Data: {min_date.date()} through {max_date.date()}
OOS starts: {oos_start.date()} (last {OOS_YEARS} years)
Round-trip trading cost assumption: {ROUND_TRIP_COST_BPS:.1f} bps

FIXED STOCK SELECTION
- Daily Close > EMA50 and EMA200
- Daily EMA20 > EMA50 > EMA200
- 6-month return >= 30%
- Close >= 50% above prior 52-week low
- Prior 20-day average turnover >= Rs 5 crore
- Prior thrust >= 30% with at least two >=2x-volume days
- Pullback after thrust between 5% and 30%

ENTRY
- Low price range, low ATR, low price volatility, volume dry-up and low volume volatility are tested in multiple variants
- Signal requires a close above the prior base high
- Entry is next trading day's open
- Structural stop is prior base low
- Entry is rejected if structural risk exceeds the entry-config cap (4%, 5% or 6%)

EXITS TESTED
- EMA10/EMA20/EMA50 close exits
- Chandelier ATR trails at 2.0/2.5/3.0/3.5 ATR
- 5/10/20-day Donchian low trails
- Supertrend 10-period at 2.5/3.0/3.5 multipliers

EXECUTION
- One open trade per symbol per strategy combination
- Re-entry only after exit and a fresh later setup
- No same-candle re-entry
- All entries are next-session open
- Gap through a known stop exits at actual session open
- Shortlist is OOS-weighted; final selection still requires reviewing concentration, yearly consistency and parameter stability
- Historical universe can still contain survivorship bias because it originates from the broker's available symbol universe
""", encoding="utf-8")
    print(f"Completed trades across all combinations: {len(trades):,}")
    print("\nTOP ROBUST CANDIDATES")
    if recommended.empty:
        print("No shortlist available; inspect comparison/stability CSVs.")
    else:
        cols = [c for c in ["EntryConfig", "ExitStrategy", "RobustnessScore", "Train_Trades", "OOS_Trades", "Train_GeoMeanPerTrade_pct", "OOS_GeoMeanPerTrade_pct", "OOS_WinRate_pct", "OOS_AvgHoldingSessions", "OOS_AvgProfitGiveback_pct", "OOS_MedianTrendMonetisationRatio"] if c in recommended.columns]
        print(recommended[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
