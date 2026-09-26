"""
EMA Active Leader Swing Backtest v2
===================================
Daily NSE cash-equity research engine using nse_6yr_historical.parquet.

Core idea
---------
A stock must first EARN Active Leader status using the user's strict trend/momentum/
volume rules. After that, the full original thrust is NOT required again for every
trade. While leader status remains valid, the engine repeatedly looks for fresh low-
volatility / low-volume contractions and tests early entries, mini-breakouts and full
base breakouts with tight 4-6% structural risk.

Execution discipline
--------------------
- All trend moving averages are DAILY EMA only.
- Signals use information available by the close of day D.
- Entry is the next trading day's actual open.
- Gap through a known stop exits at the actual open.
- One open trade per symbol per strategy combination.
- Re-entry requires a later fresh signal and a cooldown; no same-candle re-entry.
- Last two years are reserved as OOS by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

INPUT = Path("nse_6yr_historical.parquet")
OUT_SETUPS = Path("active_leader_setups.csv")
OUT_TRADES = Path("active_leader_trades.csv")
OUT_COMPARE = Path("active_leader_strategy_comparison.csv")
OUT_YEARLY = Path("active_leader_yearly.csv")
OUT_SYMBOL = Path("active_leader_symbol_results.csv")
OUT_STABILITY = Path("active_leader_parameter_stability.csv")
OUT_RECOMMENDED = Path("active_leader_recommended_candidates.csv")
OUT_LEADER_DAYS = Path("active_leader_days.csv")
OUT_NOTES = Path("active_leader_notes.txt")

LOOKBACK_6M = 126
LOOKBACK_52W = 252
MIN_HISTORY = 260
MIN_MOMENTUM_6M = 0.30
MIN_ABOVE_52W_LOW = 0.50
MIN_TURNOVER_20D = 5e7
MIN_PRIOR_THRUST = 0.30
MIN_2X_VOLUME_DAYS = 2
THRUST_LOOKBACK = 126
LEADER_GRACE_DAYS = 60
DEACTIVATE_BELOW_EMA50_DAYS = 5
REENTRY_COOLDOWN = 3
OOS_YEARS = int(os.environ.get("OOS_YEARS", "2"))
ROUND_TRIP_COST_BPS = float(os.environ.get("ROUND_TRIP_COST_BPS", "20"))


@dataclass(frozen=True)
class EntryConfig:
    name: str
    trigger: str
    base_bars: int
    trigger_bars: int
    max_range: float
    max_atr_pct: float
    max_vol_dry_ratio: float
    max_price_vol: float
    max_volume_cv: float
    breakout_vol_min: float
    stop_cap: float
    near_support_pct: float = 0.05


ENTRY_CONFIGS: Tuple[EntryConfig, ...] = (
    EntryConfig("CONTRACTION_4PCT", "contraction", 10, 0, 0.070, 0.0325, 0.70, 0.0225, 0.80, 0.00, 0.04, 0.050),
    EntryConfig("CONTRACTION_5PCT", "contraction", 10, 0, 0.080, 0.0350, 0.75, 0.0250, 0.90, 0.00, 0.05, 0.060),
    EntryConfig("MINI_3D_4PCT", "breakout", 10, 3, 0.080, 0.0350, 0.75, 0.0250, 0.90, 0.00, 0.04),
    EntryConfig("MINI_5D_5PCT", "breakout", 10, 5, 0.090, 0.0375, 0.80, 0.0275, 0.95, 0.00, 0.05),
    EntryConfig("BASE_10D_6PCT", "breakout", 10, 10, 0.100, 0.0400, 0.80, 0.0300, 1.00, 0.00, 0.06),
    EntryConfig("BASE_10D_VOL_6PCT", "breakout", 10, 10, 0.100, 0.0400, 0.80, 0.0300, 1.00, 1.20, 0.06),
)


@dataclass(frozen=True)
class ExitConfig:
    name: str
    kind: str
    p1: float
    p2: float = 0.0
    p3: float = 0.0


EXIT_CONFIGS: Tuple[ExitConfig, ...] = (
    ExitConfig("PROTECT_FAST", "state", 0.75, 1.50, 1.50),
    ExitConfig("PROTECT_BALANCED", "state", 1.00, 2.00, 2.00),
    ExitConfig("PROTECT_LOOSE", "state", 1.25, 2.50, 2.50),
    ExitConfig("EMA10_AFTER_1R", "ema_after_r", 1.00, 10.0),
    ExitConfig("CHAND_2ATR", "chandelier", 2.0),
    ExitConfig("DONCHIAN_5D", "donchian", 5.0),
)


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
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


def gt(df: pd.DataFrame, col: str, func):
    return df.groupby("Symbol", group_keys=False)[col].transform(func)


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_input(raw)
    g = df.groupby("Symbol", group_keys=False)
    df["Bar"] = g.cumcount()
    df["HistoryDays"] = df["Bar"] + 1
    for span in (10, 20, 50, 200):
        df[f"EMA{span}"] = gt(df, "Close", lambda x, s=span: x.ewm(span=s, adjust=False, min_periods=s).mean())

    df["Turnover"] = df["Close"] * df["Volume"]
    df["ADV20_PRIOR"] = gt(df, "Turnover", lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    df["RET_6M"] = gt(df, "Close", lambda x: x / x.shift(LOOKBACK_6M) - 1.0)
    df["LOW_52W_PRIOR"] = gt(df, "Low", lambda x: x.shift(1).rolling(LOOKBACK_52W, min_periods=LOOKBACK_52W).min())
    df["ABOVE_52W_LOW"] = df["Close"] / df["LOW_52W_PRIOR"] - 1.0

    prev = g["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR14"] = tr.groupby(df["Symbol"]).transform(lambda x: x.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    df["ATR_PCT"] = df["ATR14"] / df["Close"]
    df["ATR_PCT_5"] = gt(df, "ATR_PCT", lambda x: x.rolling(5, min_periods=5).mean())
    df["VOL50_PRIOR"] = gt(df, "Volume", lambda x: x.shift(1).rolling(50, min_periods=50).mean())
    df["DAY_VOL_RATIO"] = df["Volume"] / df["VOL50_PRIOR"]
    df["RET1"] = g["Close"].pct_change()

    for w in (20, 40, 60, 90):
        df[f"RET_{w}D"] = gt(df, "Close", lambda x, p=w: x / x.shift(p) - 1.0)
    df["THRUST_POINT"] = df[["RET_20D", "RET_40D", "RET_60D", "RET_90D"]].max(axis=1)
    df["HI_VOL_2X"] = (df["DAY_VOL_RATIO"] >= 2.0).astype(float)
    df["PRIOR_THRUST"] = gt(df, "THRUST_POINT", lambda x: x.shift(1).rolling(THRUST_LOOKBACK, min_periods=40).max())
    df["PRIOR_2XVOL_DAYS"] = gt(df, "HI_VOL_2X", lambda x: x.shift(1).rolling(THRUST_LOOKBACK, min_periods=40).sum())

    for n in (3, 5, 10):
        df[f"PRIOR_HIGH_{n}"] = gt(df, "High", lambda x, k=n: x.shift(1).rolling(k, min_periods=k).max())
    for n in (5, 10, 20):
        df[f"PRIOR_LOW_{n}"] = gt(df, "Low", lambda x, k=n: x.shift(1).rolling(k, min_periods=k).min())
        df[f"DONCH_{n}"] = gt(df, "Low", lambda x, k=n: x.rolling(k, min_periods=k).min())

    df["BASE_HIGH_10"] = gt(df, "High", lambda x: x.shift(1).rolling(10, min_periods=10).max())
    df["BASE_LOW_10"] = gt(df, "Low", lambda x: x.shift(1).rolling(10, min_periods=10).min())
    df["BASE_RANGE_10"] = df["BASE_HIGH_10"] / df["BASE_LOW_10"] - 1.0
    df["BASE_VOL_AVG_10"] = gt(df, "Volume", lambda x: x.shift(1).rolling(10, min_periods=10).mean())
    df["BASE_VOL_STD_10"] = gt(df, "Volume", lambda x: x.shift(1).rolling(10, min_periods=10).std(ddof=0))
    df["VOL_DRY_10"] = df["BASE_VOL_AVG_10"] / df["VOL50_PRIOR"]
    df["VOL_CV_10"] = df["BASE_VOL_STD_10"] / df["BASE_VOL_AVG_10"]
    df["PRICE_VOL_10"] = gt(df, "RET1", lambda x: x.shift(1).rolling(10, min_periods=10).std(ddof=0))

    df["NextDate"] = g["Date"].shift(-1)
    df["NextOpen"] = g["Open"].shift(-1)
    return df


def initial_leader_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["HistoryDays"] >= MIN_HISTORY)
        & (df["ADV20_PRIOR"] >= MIN_TURNOVER_20D)
        & (df["Close"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["EMA20"] > df["EMA50"])
        & (df["EMA50"] > df["EMA200"])
        & (df["RET_6M"] >= MIN_MOMENTUM_6M)
        & (df["ABOVE_52W_LOW"] >= MIN_ABOVE_52W_LOW)
        & (df["PRIOR_THRUST"] >= MIN_PRIOR_THRUST)
        & (df["PRIOR_2XVOL_DAYS"] >= MIN_2X_VOLUME_DAYS)
    )


def add_active_leader_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seed = initial_leader_mask(df)
    active = np.zeros(len(df), dtype=bool)
    age = np.full(len(df), np.nan)
    seed_flag = seed.to_numpy(bool)

    for _, idx in df.groupby("Symbol", sort=False).groups.items():
        is_active = False
        bars_since_seed = 10**9
        below50_streak = 0
        for pos in list(idx):
            r = df.loc[pos]
            if seed_flag[pos]:
                is_active = True
                bars_since_seed = 0
                below50_streak = 0
            elif is_active:
                bars_since_seed += 1
                if pd.notna(r["EMA50"]) and r["Close"] < r["EMA50"]:
                    below50_streak += 1
                else:
                    below50_streak = 0

                hard_break = (
                    (pd.notna(r["EMA200"]) and r["Close"] < r["EMA200"])
                    or (pd.notna(r["EMA50"]) and pd.notna(r["EMA200"]) and r["EMA50"] <= r["EMA200"])
                    or below50_streak >= DEACTIVATE_BELOW_EMA50_DAYS
                )
                stale = bars_since_seed > LEADER_GRACE_DAYS and (
                    pd.isna(r["RET_6M"]) or r["RET_6M"] < 0.10
                )
                if hard_break or stale:
                    is_active = False
            active[pos] = is_active
            age[pos] = bars_since_seed if is_active else np.nan

    df["InitialLeaderSeed"] = seed
    df["ActiveLeader"] = active
    df["LeaderAge"] = age
    return df


def build_setups(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df["ActiveLeader"]
        & (df["ADV20_PRIOR"] >= MIN_TURNOVER_20D)
        & df["NextOpen"].notna()
        & df["BASE_LOW_10"].notna()
        & df["BASE_HIGH_10"].notna()
        & df["ATR_PCT_5"].notna()
        & df["VOL_DRY_10"].notna()
        & df["VOL_CV_10"].notna()
        & df["PRICE_VOL_10"].notna()
    )
    rows: List[pd.DataFrame] = []

    for cfg in ENTRY_CONFIGS:
        stop = df["BASE_LOW_10"] * 0.995
        entry = df["NextOpen"]
        risk = (entry - stop) / entry
        contraction = (
            (df["BASE_RANGE_10"] <= cfg.max_range)
            & (df["ATR_PCT_5"] <= cfg.max_atr_pct)
            & (df["VOL_DRY_10"] <= cfg.max_vol_dry_ratio)
            & (df["PRICE_VOL_10"] <= cfg.max_price_vol)
            & (df["VOL_CV_10"] <= cfg.max_volume_cv)
        )

        if cfg.trigger == "contraction":
            trigger = df["Close"] <= df["BASE_LOW_10"] * (1.0 + cfg.near_support_pct)
        else:
            trigger = (
                (df["Close"] > df[f"PRIOR_HIGH_{cfg.trigger_bars}"])
                & (df["DAY_VOL_RATIO"] >= cfg.breakout_vol_min)
            )

        mask = common & contraction & trigger & (entry > stop) & (risk > 0) & (risk <= cfg.stop_cap)
        cols = [
            "Symbol", "Date", "Bar", "Close", "NextDate", "NextOpen",
            "EMA20", "EMA50", "EMA200", "RET_6M", "ABOVE_52W_LOW",
            "ADV20_PRIOR", "LeaderAge", "DAY_VOL_RATIO",
        ]
        s = df.loc[mask, cols].copy()
        if s.empty:
            continue
        s["EntryConfig"] = cfg.name
        s["Trigger"] = cfg.trigger
        s["BreakoutLevel"] = np.where(
            cfg.trigger == "contraction",
            df.loc[mask, "BASE_HIGH_10"].to_numpy(),
            df.loc[mask, f"PRIOR_HIGH_{cfg.trigger_bars}"].to_numpy(),
        )
        s["StructuralStop"] = stop.loc[mask].to_numpy()
        s["InitialRisk_pct"] = risk.loc[mask].to_numpy() * 100
        s["BaseRange_pct"] = df.loc[mask, "BASE_RANGE_10"].to_numpy() * 100
        s["VolDryRatio"] = df.loc[mask, "VOL_DRY_10"].to_numpy()
        s["PriceVol_pct"] = df.loc[mask, "PRICE_VOL_10"].to_numpy() * 100
        s["VolumeCV"] = df.loc[mask, "VOL_CV_10"].to_numpy()
        s["SetupScore"] = (
            (cfg.stop_cap * 100 - s["InitialRisk_pct"]).clip(lower=0) * 3
            + (cfg.max_range * 100 - s["BaseRange_pct"]).clip(lower=0) * 1.5
            + (1 - s["VolDryRatio"].clip(0, 1)) * 12
            + (1 - s["VolumeCV"].clip(0, 1.5) / 1.5) * 5
        )
        rows.append(s)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"Date": "SignalDate", "NextDate": "EntryDate", "NextOpen": "EntryPrice"}
    )
    out["EntryBar"] = out["Bar"].astype(int) + 1
    return out.sort_values(["EntryConfig", "Symbol", "EntryDate", "SetupScore"], ascending=[True, True, True, False]).reset_index(drop=True)


def stop_hit(op: float, lo: float, stop: float):
    if not np.isfinite(stop) or stop <= 0:
        return False, np.nan, ""
    if op <= stop:
        return True, float(op), "GAP_THROUGH_STOP"
    if lo <= stop:
        return True, float(stop), "STOP"
    return False, np.nan, ""


def simulate_trade(sg: pd.DataFrame, entry_bar: int, entry_price: float, initial_stop: float, cfg: ExitConfig) -> Dict[str, object]:
    last_bar = int(sg.index.max())
    if entry_bar not in sg.index:
        return {}
    risk_rupees = entry_price - initial_stop
    if risk_rupees <= 0:
        return {}

    highest = entry_price
    trail = initial_stop
    mfe = -np.inf
    mae = np.inf
    max_r = 0.0
    exit_bar, exit_price, exit_reason = last_bar, float(sg.loc[last_bar, "Close"]), "END_OF_DATA"

    for bar in range(entry_bar, last_bar + 1):
        if bar not in sg.index:
            continue
        r = sg.loc[bar]
        op, hi, lo, cl = (float(r[x]) for x in ["Open", "High", "Low", "Close"])

        hit, px, reason = stop_hit(op, lo, trail)
        mfe = max(mfe, hi / entry_price - 1.0)
        mae = min(mae, lo / entry_price - 1.0)
        if hit:
            exit_bar, exit_price = bar, px
            exit_reason = reason if trail <= initial_stop + 1e-12 else f"{cfg.name}_{reason}"
            break

        highest = max(highest, hi)
        max_r = max(max_r, (highest - entry_price) / risk_rupees)

        if cfg.kind == "state":
            first_r, second_r, atr_mult = cfg.p1, cfg.p2, cfg.p3
            atr = float(r["ATR14"])
            if max_r >= first_r:
                trail = max(trail, entry_price)
            if max_r >= second_r:
                trail = max(trail, entry_price + 0.50 * risk_rupees)
            if max_r >= second_r + 1.0 and np.isfinite(atr):
                trail = max(trail, highest - atr_mult * atr, entry_price + risk_rupees)

        elif cfg.kind == "ema_after_r":
            activate_r, ema_span = cfg.p1, int(cfg.p2)
            if max_r >= activate_r:
                trail = max(trail, entry_price)
                ema = float(r[f"EMA{ema_span}"])
                if np.isfinite(ema) and cl < ema:
                    nb = bar + 1
                    if nb in sg.index:
                        exit_bar, exit_price, exit_reason = nb, float(sg.loc[nb, "Open"]), cfg.name
                    else:
                        exit_bar, exit_price, exit_reason = bar, cl, f"{cfg.name}_EOD"
                    break

        elif cfg.kind == "chandelier":
            atr = float(r["ATR14"])
            if np.isfinite(atr):
                trail = max(trail, highest - cfg.p1 * atr)

        elif cfg.kind == "donchian":
            n = int(cfg.p1)
            v = float(r[f"DONCH_{n}"])
            if np.isfinite(v):
                trail = max(trail, v)

    gross = exit_price / entry_price - 1.0
    net = gross - ROUND_TRIP_COST_BPS / 10000.0
    rr = gross * entry_price / risk_rupees
    er = sg.loc[exit_bar]
    return {
        "ExitBar": int(exit_bar),
        "ExitDate": pd.Timestamp(er["Date"]),
        "ExitPrice": float(exit_price),
        "ExitReason": exit_reason,
        "GrossReturn_pct": gross * 100,
        "NetReturn_pct": net * 100,
        "RealizedR": rr,
        "MaxR": max_r,
        "MFE_pct": mfe * 100 if np.isfinite(mfe) else np.nan,
        "MAE_pct": mae * 100 if np.isfinite(mae) else np.nan,
        "ProfitGiveback_pct": max(0.0, mfe - gross) * 100 if np.isfinite(mfe) else np.nan,
        "HoldingSessions": int(exit_bar - entry_bar + 1),
    }


def run_backtests(df: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()
    frames = {
        sym: g.sort_values("Bar").set_index("Bar", drop=False)
        for sym, g in df.groupby("Symbol", sort=False)
    }
    trades: List[Dict[str, object]] = []

    for entry_name, es in setups.groupby("EntryConfig", sort=False):
        for exit_cfg in EXIT_CONFIGS:
            for sym, ss in es.groupby("Symbol", sort=False):
                sg = frames.get(sym)
                if sg is None:
                    continue
                last_exit = -10**9
                used_entry_bars = set()
                for _, setup in ss.sort_values(["EntryBar", "SetupScore"], ascending=[True, False]).iterrows():
                    eb = int(setup["EntryBar"])
                    if eb in used_entry_bars:
                        continue
                    used_entry_bars.add(eb)
                    if eb <= last_exit + REENTRY_COOLDOWN:
                        continue
                    result = simulate_trade(
                        sg,
                        eb,
                        float(setup["EntryPrice"]),
                        float(setup["StructuralStop"]),
                        exit_cfg,
                    )
                    if not result:
                        continue
                    rec = {
                        "EntryConfig": entry_name,
                        "ExitStrategy": exit_cfg.name,
                        "Symbol": sym,
                        "SignalDate": setup["SignalDate"],
                        "EntryDate": setup["EntryDate"],
                        "EntryBar": eb,
                        "EntryPrice": float(setup["EntryPrice"]),
                        "StructuralStop": float(setup["StructuralStop"]),
                        "InitialRisk_pct": float(setup["InitialRisk_pct"]),
                        "SetupScore": float(setup["SetupScore"]),
                        "Trigger": setup["Trigger"],
                        "LeaderAge": float(setup["LeaderAge"]),
                        "BaseRange_pct": float(setup["BaseRange_pct"]),
                        "VolDryRatio": float(setup["VolDryRatio"]),
                        "PriceVol_pct": float(setup["PriceVol_pct"]),
                        "VolumeCV": float(setup["VolumeCV"]),
                    }
                    rec.update(result)
                    trades.append(rec)
                    last_exit = int(result["ExitBar"])

    if not trades:
        return pd.DataFrame()
    out = pd.DataFrame(trades)
    out["EntryDate"] = pd.to_datetime(out["EntryDate"])
    out["ExitDate"] = pd.to_datetime(out["ExitDate"])
    return out.sort_values(["EntryConfig", "ExitStrategy", "EntryDate", "Symbol"]).reset_index(drop=True)


def profit_factor(x: pd.Series) -> float:
    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses <= 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def summary_metrics(t: pd.DataFrame, label: str) -> Dict[str, object]:
    if t.empty:
        return {"Period": label, "Trades": 0}
    r = t["NetReturn_pct"].dropna()
    return {
        "Period": label,
        "Trades": len(t),
        "Symbols": t["Symbol"].nunique(),
        "WinRate_pct": (r > 0).mean() * 100,
        "MeanNetReturn_pct": r.mean(),
        "MedianNetReturn_pct": r.median(),
        "GeoMeanPerTrade_pct": (np.exp(np.log1p(r / 100).mean()) - 1) * 100,
        "ProfitFactor": profit_factor(r),
        "AvgHoldingSessions": t["HoldingSessions"].mean(),
        "MedianHoldingSessions": t["HoldingSessions"].median(),
        "AvgInitialRisk_pct": t["InitialRisk_pct"].mean(),
        "AvgRealizedR": t["RealizedR"].mean(),
        "MedianRealizedR": t["RealizedR"].median(),
        "AvgMaxR": t["MaxR"].mean(),
        "AvgMFE_pct": t["MFE_pct"].mean(),
        "AvgMAE_pct": t["MAE_pct"].mean(),
        "AvgProfitGiveback_pct": t["ProfitGiveback_pct"].mean(),
        "PctTrades_ge_10": (r >= 10).mean() * 100,
        "PctTrades_ge_20": (r >= 20).mean() * 100,
        "PctTrades_le_minus5": (r <= -5).mean() * 100,
        "ReturnPerHoldingDay_bp": (r / t["HoldingSessions"].clip(lower=1)).mean() * 100,
    }


def build_symbol_results(trades: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for period, mask in (
        ("ALL", pd.Series(True, index=trades.index)),
        ("TRAIN", trades["EntryDate"] < oos_start),
        ("OOS", trades["EntryDate"] >= oos_start),
    ):
        sub = trades.loc[mask].copy()
        for (ec, ex, sym), t in sub.groupby(["EntryConfig", "ExitStrategy", "Symbol"]):
            t = t.sort_values("EntryDate")
            growth = np.prod(1 + t["NetReturn_pct"].to_numpy(float) / 100.0) - 1.0
            underlying = float(t.iloc[-1]["ExitPrice"]) / float(t.iloc[0]["EntryPrice"]) - 1.0
            rows.append({
                "Period": period,
                "EntryConfig": ec,
                "ExitStrategy": ex,
                "Symbol": sym,
                "Trades": len(t),
                "FirstEntryDate": t.iloc[0]["EntryDate"],
                "LastExitDate": t.iloc[-1]["ExitDate"],
                "StrategyCompoundedReturn_pct": growth * 100,
                "UnderlyingFirstEntryToLastExit_pct": underlying * 100,
                "TrendMonetisationRatio": growth / underlying if underlying > 0 else np.nan,
                "WinRate_pct": (t["NetReturn_pct"] > 0).mean() * 100,
                "AvgHoldingSessions": t["HoldingSessions"].mean(),
                "AvgGiveback_pct": t["ProfitGiveback_pct"].mean(),
            })
    return pd.DataFrame(rows)


def build_comparison(trades: pd.DataFrame, symbol_results: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (ec, ex), t in trades.groupby(["EntryConfig", "ExitStrategy"]):
        for label, sub in (
            ("ALL", t),
            ("TRAIN", t[t["EntryDate"] < oos_start]),
            ("OOS", t[t["EntryDate"] >= oos_start]),
        ):
            rec = {"EntryConfig": ec, "ExitStrategy": ex}
            rec.update(summary_metrics(sub, label))
            sr = symbol_results[
                (symbol_results["Period"] == label)
                & (symbol_results["EntryConfig"] == ec)
                & (symbol_results["ExitStrategy"] == ex)
            ]
            if not sr.empty:
                rec["MedianSymbolStrategyReturn_pct"] = sr["StrategyCompoundedReturn_pct"].median()
                positive_under = sr[sr["UnderlyingFirstEntryToLastExit_pct"] > 0]
                rec["MedianTrendMonetisationRatio"] = positive_under["TrendMonetisationRatio"].median() if not positive_under.empty else np.nan
                rec["PctSymbolsTrendMonetisation_gt_1"] = (positive_under["TrendMonetisationRatio"] > 1).mean() * 100 if not positive_under.empty else np.nan
                rec["PctSymbolsTrendMonetisation_gt_1_5"] = (positive_under["TrendMonetisationRatio"] > 1.5).mean() * 100 if not positive_under.empty else np.nan
                big = positive_under[positive_under["UnderlyingFirstEntryToLastExit_pct"] >= 100]
                rec["BigWinnerSymbols"] = len(big)
                rec["BigWinnerMedianMonetisation"] = big["TrendMonetisationRatio"].median() if not big.empty else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def build_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    x = trades.copy()
    x["Year"] = x["EntryDate"].dt.year
    rows = []
    for (ec, ex, year), t in x.groupby(["EntryConfig", "ExitStrategy", "Year"]):
        rec = {"EntryConfig": ec, "ExitStrategy": ex, "Year": int(year)}
        rec.update(summary_metrics(t, str(year)))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_stability(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    metric_cols = [
        "Trades", "WinRate_pct", "MeanNetReturn_pct", "MedianNetReturn_pct",
        "GeoMeanPerTrade_pct", "ProfitFactor", "AvgHoldingSessions", "AvgRealizedR",
        "MedianRealizedR", "AvgProfitGiveback_pct", "ReturnPerHoldingDay_bp",
        "MedianTrendMonetisationRatio", "PctSymbolsTrendMonetisation_gt_1",
        "PctSymbolsTrendMonetisation_gt_1_5", "BigWinnerSymbols", "BigWinnerMedianMonetisation",
    ]
    keys = ["EntryConfig", "ExitStrategy"]
    train = comparison[comparison["Period"] == "TRAIN"][keys + [c for c in metric_cols if c in comparison.columns]].copy()
    oos = comparison[comparison["Period"] == "OOS"][keys + [c for c in metric_cols if c in comparison.columns]].copy()
    train = train.rename(columns={c: f"Train_{c}" for c in metric_cols if c in train.columns})
    oos = oos.rename(columns={c: f"OOS_{c}" for c in metric_cols if c in oos.columns})
    s = train.merge(oos, on=keys, how="outer")
    if "Train_GeoMeanPerTrade_pct" in s and "OOS_GeoMeanPerTrade_pct" in s:
        s["GeoMean_OOS_minus_Train_pct"] = s["OOS_GeoMeanPerTrade_pct"] - s["Train_GeoMeanPerTrade_pct"]
    if "Train_ProfitFactor" in s and "OOS_ProfitFactor" in s:
        s["PF_OOS_minus_Train"] = s["OOS_ProfitFactor"] - s["Train_ProfitFactor"]
    return s


def build_recommended(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    s = stability.copy()
    eligible = s[(s.get("Train_Trades", 0) >= 40) & (s.get("OOS_Trades", 0) >= 20)].copy()
    if eligible.empty:
        eligible = s.copy()

    def col(frame, name, default=0.0):
        if name in frame.columns:
            return frame[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    eligible["RobustnessScore"] = (
        col(eligible, "OOS_GeoMeanPerTrade_pct", -99) * 3.0
        + col(eligible, "OOS_ProfitFactor", 0) * 1.5
        + col(eligible, "OOS_MedianRealizedR", -5) * 1.0
        + col(eligible, "OOS_ReturnPerHoldingDay_bp", -100) * 0.015
        + col(eligible, "OOS_MedianTrendMonetisationRatio", 0) * 2.0
        + col(eligible, "OOS_BigWinnerMedianMonetisation", 0) * 2.0
        - col(eligible, "OOS_AvgProfitGiveback_pct", 20) * 0.12
        - col(eligible, "GeoMean_OOS_minus_Train_pct", 0).abs() * 0.75
    )
    return eligible.sort_values("RobustnessScore", ascending=False).head(25)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing historical database: {INPUT}")

    print("Loading historical NSE database...")
    raw = pd.read_parquet(INPUT)
    df = prepare_features(raw)
    df = add_active_leader_state(df)
    min_date = pd.Timestamp(df["Date"].min())
    max_date = pd.Timestamp(df["Date"].max())
    oos_start = max_date - pd.DateOffset(years=OOS_YEARS)
    print(f"Rows={len(df):,}; symbols={df['Symbol'].nunique():,}; data={min_date.date()} to {max_date.date()}")
    print(f"Initial leader seeds={int(df['InitialLeaderSeed'].sum()):,}; active-leader days={int(df['ActiveLeader'].sum()):,}")

    leader_days = df[df["ActiveLeader"]][[
        "Symbol", "Date", "Close", "EMA20", "EMA50", "EMA200", "RET_6M",
        "ABOVE_52W_LOW", "ADV20_PRIOR", "LeaderAge", "InitialLeaderSeed"
    ]].copy()
    leader_days.to_csv(OUT_LEADER_DAYS, index=False)

    setups = build_setups(df)
    print(f"Fresh low-volatility setup/entry observations: {len(setups):,}")
    setups.to_csv(OUT_SETUPS, index=False)

    if setups.empty:
        for p in [OUT_TRADES, OUT_COMPARE, OUT_YEARLY, OUT_SYMBOL, OUT_STABILITY, OUT_RECOMMENDED]:
            pd.DataFrame().to_csv(p, index=False)
        OUT_NOTES.write_text("No qualifying Active Leader setups were found.\n", encoding="utf-8")
        return

    print(f"Running {len(ENTRY_CONFIGS)} entry variants x {len(EXIT_CONFIGS)} exit variants...")
    trades = run_backtests(df, setups)
    trades.to_csv(OUT_TRADES, index=False)

    symbol = build_symbol_results(trades, oos_start)
    comparison = build_comparison(trades, symbol, oos_start)
    yearly = build_yearly(trades)
    stability = build_stability(comparison)
    recommended = build_recommended(stability)

    symbol.to_csv(OUT_SYMBOL, index=False)
    comparison.to_csv(OUT_COMPARE, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    stability.to_csv(OUT_STABILITY, index=False)
    recommended.to_csv(OUT_RECOMMENDED, index=False)

    OUT_NOTES.write_text(f"""EMA ACTIVE LEADER SWING BACKTEST v2

Data: {min_date.date()} through {max_date.date()}
OOS starts: {oos_start.date()} (last {OOS_YEARS} years)
Round-trip cost assumption: {ROUND_TRIP_COST_BPS:.1f} bps

INITIAL LEADER QUALIFICATION
- Daily Close > EMA50 and EMA200
- Daily EMA20 > EMA50 > EMA200
- 6-month return >= 30%
- Close >= 50% above prior 52-week low
- Prior 20-day average turnover >= Rs 5 crore
- Prior 126-session thrust >= 30%
- At least two prior >=2x-volume days

ACTIVE LEADER STATE
- Full original qualification is NOT required again for every trade.
- Once seeded, leader status persists through later consolidations.
- It is deactivated after a material trend break: five consecutive closes below EMA50,
  Close below EMA200, or EMA50 <= EMA200.
- A stale leader can also expire after the grace window if six-month momentum falls below 10%.

RE-ENTRY / SETUPS
- 10-day low-volatility, low-volume contraction is the reusable setup.
- Tested entries: inside contraction near support, 3-day mini-breakout, 5-day mini-breakout,
  10-day base breakout, and 10-day breakout with volume confirmation.
- Structural stop is just below the prior 10-day base low.
- Entry is rejected when risk exceeds the configuration cap (4%, 5%, or 6%).
- Re-entry is allowed after exit with a {REENTRY_COOLDOWN}-session cooldown and a fresh later signal.

EXITS
- PROTECT_FAST / BALANCED / LOOSE: profit-state exits using R milestones + ATR trailing.
- EMA10_AFTER_1R: initial structural stop, breakeven after +1R, then EMA10 close exit.
- CHAND_2ATR and DONCHIAN_5D retained as controls.

REPORTING FIX
- Trend monetisation is now calculated independently for ALL, TRAIN and OOS.
- Big-winner monetisation explicitly reports stocks that rose >=100% over the strategy's
  first-entry to last-exit interval inside each period.
""", encoding="utf-8")

    print(f"Completed trades across all combinations: {len(trades):,}")
    print("\nTOP ROBUST CANDIDATES")
    if recommended.empty:
        print("No shortlist available; inspect comparison/stability files.")
    else:
        cols = [c for c in [
            "EntryConfig", "ExitStrategy", "RobustnessScore",
            "Train_Trades", "OOS_Trades", "Train_GeoMeanPerTrade_pct",
            "OOS_GeoMeanPerTrade_pct", "OOS_ProfitFactor", "OOS_WinRate_pct",
            "OOS_AvgHoldingSessions", "OOS_AvgProfitGiveback_pct",
            "OOS_MedianTrendMonetisationRatio", "OOS_BigWinnerMedianMonetisation",
        ] if c in recommended.columns]
        print(recommended[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
