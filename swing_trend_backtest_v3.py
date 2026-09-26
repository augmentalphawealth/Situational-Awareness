"""
Strict Dry-Up Active Leader Backtest v3
=======================================
Research focus:
- Daily EMA trend MUST still be intact on every entry signal day.
- Volume dry-up means not only lower average volume, but very low volume volatility.
- Dry-up can be tied explicitly to a prior high-volume expansion.
- Entry timing is tested earlier: reversal day, 2-day high, 3-day high, full base high.
- Market breadth Composite_Score gates are tested as variants; none is assumed in advance.
- Train/OOS, big-winner monetisation and winner concentration are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import swing_trend_backtest_v2 as core

INPUT = Path("nse_6yr_historical.parquet")
BREADTH = Path("historical_breadth_regime_6yr.csv")

OUT_SETUPS = Path("strict_dryup_v3_setups.csv")
OUT_TRADES = Path("strict_dryup_v3_trades.csv")
OUT_COMPARE = Path("strict_dryup_v3_strategy_comparison.csv")
OUT_YEARLY = Path("strict_dryup_v3_yearly.csv")
OUT_SYMBOL = Path("strict_dryup_v3_symbol_results.csv")
OUT_STABILITY = Path("strict_dryup_v3_parameter_stability.csv")
OUT_RECOMMENDED = Path("strict_dryup_v3_recommended_candidates.csv")
OUT_CONCENTRATION = Path("strict_dryup_v3_concentration.csv")
OUT_NOTES = Path("strict_dryup_v3_notes.txt")


@dataclass(frozen=True)
class V3Entry:
    name: str
    trigger: str
    risk_cap: float
    regime_min: int | None = None
    require_recent_expansion: bool = False
    max_dry_vs_expansion: float = 1.0


ENTRY_CONFIGS = (
    V3Entry("REVERSAL_DRY_4PCT", "reversal", 0.04),
    V3Entry("HIGH2_DRY_4PCT", "high2", 0.04),
    V3Entry("HIGH3_DRY_4PCT", "high3", 0.04),
    V3Entry("BASE10_DRY_4PCT", "base10", 0.04),
    V3Entry("HIGH2_DRY_REG40", "high2", 0.04, regime_min=40),
    V3Entry("HIGH2_DRY_REG50", "high2", 0.04, regime_min=50),
    V3Entry("HIGH2_DRY_EXP", "high2", 0.04, require_recent_expansion=True, max_dry_vs_expansion=0.60),
    V3Entry("HIGH2_DRY_EXP_REG40", "high2", 0.04, regime_min=40, require_recent_expansion=True, max_dry_vs_expansion=0.60),
    V3Entry("HIGH3_DRY_EXP_REG40", "high3", 0.04, regime_min=40, require_recent_expansion=True, max_dry_vs_expansion=0.60),
)

V3_EXITS = (
    core.ExitConfig("CHAND_2ATR", "chandelier", 2.0),
    core.ExitConfig("CHAND_2_5ATR", "chandelier", 2.5),
    core.ExitConfig("PROTECT_FAST", "state", 0.75, 1.50, 1.50),
    core.ExitConfig("PROTECT_BALANCED", "state", 1.00, 2.00, 2.00),
    core.ExitConfig("EMA10_AFTER_1R", "ema_after_r", 1.00, 10.0),
    core.ExitConfig("DONCHIAN_5D", "donchian", 5.0),
)

# User's stated setup translated literally.
MAX_BASE_RANGE = 0.07
MAX_ATR_PCT_5 = 0.0325
MAX_VOL_DRY_RATIO = 0.70
MAX_VOLUME_CV = 0.30
MAX_PRICE_VOL = 0.0225
MIN_RECENT_THRUST = 0.30
MIN_RECENT_2X_VOL_DAYS = 2


def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Symbol", group_keys=False)

    df["PrevClose"] = g["Close"].shift(1)
    df["PRIOR_HIGH_2"] = g["High"].transform(lambda x: x.shift(1).rolling(2, min_periods=2).max())

    # Compare the quiet base directly with the preceding expansion's volume.
    df["VOL5"] = g["Volume"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df["PREBASE_MAX_VOL5_60"] = df.groupby("Symbol", group_keys=False)["VOL5"].transform(
        lambda x: x.shift(10).rolling(60, min_periods=30).max()
    )
    df["DRY_VS_EXPANSION"] = df["BASE_VOL_AVG_10"] / df["PREBASE_MAX_VOL5_60"]

    df["PREBASE_THRUST_60"] = df.groupby("Symbol", group_keys=False)["THRUST_POINT"].transform(
        lambda x: x.shift(10).rolling(60, min_periods=30).max()
    )
    df["PREBASE_2XVOL_60"] = df.groupby("Symbol", group_keys=False)["HI_VOL_2X"].transform(
        lambda x: x.shift(10).rolling(60, min_periods=30).sum()
    )
    return df


def merge_market_breadth(df: pd.DataFrame) -> pd.DataFrame:
    if not BREADTH.exists():
        df = df.copy()
        df["Composite_Score"] = np.nan
        return df
    b = pd.read_csv(BREADTH, usecols=["Date", "Composite_Score"])
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce").dt.normalize()
    b["Composite_Score"] = pd.to_numeric(b["Composite_Score"], errors="coerce")
    b = b.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")
    return df.merge(b, on="Date", how="left")


def build_setups(df: pd.DataFrame) -> pd.DataFrame:
    # Unlike v2, the user's EMA trend condition is enforced at EVERY entry signal.
    trend_now = (
        (df["Close"] > df["EMA50"])
        & (df["Close"] > df["EMA200"])
        & (df["EMA20"] > df["EMA50"])
        & (df["EMA50"] > df["EMA200"])
    )

    strict_dryup = (
        (df["BASE_RANGE_10"] <= MAX_BASE_RANGE)
        & (df["ATR_PCT_5"] <= MAX_ATR_PCT_5)
        & (df["VOL_DRY_10"] <= MAX_VOL_DRY_RATIO)
        & (df["VOL_CV_10"] <= MAX_VOLUME_CV)
        & (df["PRICE_VOL_10"] <= MAX_PRICE_VOL)
    )

    common = (
        df["ActiveLeader"]
        & trend_now
        & strict_dryup
        & (df["ADV20_PRIOR"] >= core.MIN_TURNOVER_20D)
        & df["NextOpen"].notna()
        & df["BASE_LOW_10"].notna()
        & df["BASE_HIGH_10"].notna()
    )

    rows: List[pd.DataFrame] = []
    for cfg in ENTRY_CONFIGS:
        stop = df["BASE_LOW_10"] * 0.995
        entry = df["NextOpen"]
        risk = (entry - stop) / entry

        if cfg.trigger == "reversal":
            trigger = (df["Close"] > df["PrevClose"]) & (df["Close"] > df["Open"])
            trigger_level = df["Close"]
        elif cfg.trigger == "high2":
            trigger = df["Close"] > df["PRIOR_HIGH_2"]
            trigger_level = df["PRIOR_HIGH_2"]
        elif cfg.trigger == "high3":
            trigger = df["Close"] > df["PRIOR_HIGH_3"]
            trigger_level = df["PRIOR_HIGH_3"]
        elif cfg.trigger == "base10":
            trigger = df["Close"] > df["BASE_HIGH_10"]
            trigger_level = df["BASE_HIGH_10"]
        else:
            raise ValueError(f"Unknown trigger: {cfg.trigger}")

        regime = pd.Series(True, index=df.index)
        if cfg.regime_min is not None:
            regime = df["Composite_Score"].fillna(-1) >= cfg.regime_min

        recent_expansion = pd.Series(True, index=df.index)
        if cfg.require_recent_expansion:
            recent_expansion = (
                (df["PREBASE_THRUST_60"] >= MIN_RECENT_THRUST)
                & (df["PREBASE_2XVOL_60"] >= MIN_RECENT_2X_VOL_DAYS)
                & (df["DRY_VS_EXPANSION"] <= cfg.max_dry_vs_expansion)
            )

        mask = common & trigger & regime & recent_expansion & (entry > stop) & (risk > 0) & (risk <= cfg.risk_cap)

        cols = [
            "Symbol", "Date", "Bar", "Close", "NextDate", "NextOpen", "EMA20", "EMA50", "EMA200",
            "RET_6M", "ABOVE_52W_LOW", "ADV20_PRIOR", "LeaderAge", "DAY_VOL_RATIO", "Composite_Score",
            "DRY_VS_EXPANSION", "PREBASE_THRUST_60", "PREBASE_2XVOL_60",
        ]
        s = df.loc[mask, cols].copy()
        if s.empty:
            continue
        s["EntryConfig"] = cfg.name
        s["Trigger"] = cfg.trigger
        s["BreakoutLevel"] = trigger_level.loc[mask].to_numpy()
        s["StructuralStop"] = stop.loc[mask].to_numpy()
        s["InitialRisk_pct"] = risk.loc[mask].to_numpy() * 100
        s["BaseRange_pct"] = df.loc[mask, "BASE_RANGE_10"].to_numpy() * 100
        s["VolDryRatio"] = df.loc[mask, "VOL_DRY_10"].to_numpy()
        s["PriceVol_pct"] = df.loc[mask, "PRICE_VOL_10"].to_numpy() * 100
        s["VolumeCV"] = df.loc[mask, "VOL_CV_10"].to_numpy()
        s["SetupScore"] = (
            (cfg.risk_cap * 100 - s["InitialRisk_pct"]).clip(lower=0) * 3
            + (MAX_BASE_RANGE * 100 - s["BaseRange_pct"]).clip(lower=0) * 1.5
            + (MAX_VOL_DRY_RATIO - s["VolDryRatio"]).clip(lower=0) * 15
            + (MAX_VOLUME_CV - s["VolumeCV"]).clip(lower=0) * 20
        )
        rows.append(s)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"Date": "SignalDate", "NextDate": "EntryDate", "NextOpen": "EntryPrice"}
    )
    out["EntryBar"] = out["Bar"].astype(int) + 1
    return out.sort_values(
        ["EntryConfig", "Symbol", "EntryDate", "SetupScore"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def concentration_report(trades: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (ec, ex), t in trades.groupby(["EntryConfig", "ExitStrategy"]):
        for period, d in (
            ("TRAIN", t[t["EntryDate"] < oos_start]),
            ("OOS", t[t["EntryDate"] >= oos_start]),
        ):
            r = d["NetReturn_pct"].sort_values(ascending=False).reset_index(drop=True)
            if r.empty:
                continue
            positive_sum = r[r > 0].sum()
            tail5 = r.iloc[5:]
            tail10 = r.iloc[10:]
            rows.append({
                "EntryConfig": ec,
                "ExitStrategy": ex,
                "Period": period,
                "Trades": len(r),
                "Top1Return_pct": r.iloc[0],
                "Top5Sum_pct": r.head(5).sum(),
                "Top10Sum_pct": r.head(10).sum(),
                "Top10ShareGrossPositive_pct": (r.head(10).sum() / positive_sum * 100) if positive_sum > 0 else np.nan,
                "MeanReturnWithoutTop5_pct": tail5.mean() if len(tail5) else np.nan,
                "MeanReturnWithoutTop10_pct": tail10.mean() if len(tail10) else np.nan,
                "GeoMeanWithoutTop10_pct": (np.exp(np.log1p(tail10 / 100).mean()) - 1) * 100 if len(tail10) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    print("Loading historical NSE database...")
    raw = pd.read_parquet(INPUT)
    df = core.prepare_features(raw)
    df = core.add_active_leader_state(df)
    df = add_v3_features(df)
    df = merge_market_breadth(df)

    min_date = pd.Timestamp(df["Date"].min())
    max_date = pd.Timestamp(df["Date"].max())
    oos_start = max_date - pd.DateOffset(years=core.OOS_YEARS)

    print(f"Rows={len(df):,}; symbols={df['Symbol'].nunique():,}; data={min_date.date()} to {max_date.date()}")
    print(f"OOS starts={oos_start.date()}")

    setups = build_setups(df)
    setups.to_csv(OUT_SETUPS, index=False)
    print(f"Strict dry-up setup observations: {len(setups):,}")

    if setups.empty:
        for p in [OUT_TRADES, OUT_COMPARE, OUT_YEARLY, OUT_SYMBOL, OUT_STABILITY, OUT_RECOMMENDED, OUT_CONCENTRATION]:
            pd.DataFrame().to_csv(p, index=False)
        return

    # Reuse the already-audited execution engine but with v3 exit candidates.
    core.EXIT_CONFIGS = V3_EXITS
    trades = core.run_backtests(df, setups)
    trades.to_csv(OUT_TRADES, index=False)

    symbol = core.build_symbol_results(trades, oos_start)
    comparison = core.build_comparison(trades, symbol, oos_start)
    yearly = core.build_yearly(trades)
    stability = core.build_stability(comparison)
    recommended = core.build_recommended(stability)
    concentration = concentration_report(trades, oos_start)

    symbol.to_csv(OUT_SYMBOL, index=False)
    comparison.to_csv(OUT_COMPARE, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    stability.to_csv(OUT_STABILITY, index=False)
    recommended.to_csv(OUT_RECOMMENDED, index=False)
    concentration.to_csv(OUT_CONCENTRATION, index=False)

    OUT_NOTES.write_text(f"""STRICT DRY-UP ACTIVE LEADER BACKTEST v3

Data: {min_date.date()} through {max_date.date()}
OOS starts: {oos_start.date()}
Round-trip cost: {core.ROUND_TRIP_COST_BPS:.1f} bps

HARD ENTRY-DAY RULES
- Close > EMA50 and EMA200.
- EMA20 > EMA50 > EMA200.
- Stock must already be an Active Leader.
- 10-day base range <= {MAX_BASE_RANGE*100:.1f}%.
- 5-day average ATR% <= {MAX_ATR_PCT_5*100:.2f}%.
- 10-day average volume <= {MAX_VOL_DRY_RATIO*100:.0f}% of prior 50-day average.
- Volume coefficient of variation <= {MAX_VOLUME_CV:.2f}; this is the strict volume-volatility dry-up rule.
- 10-day price-return volatility <= {MAX_PRICE_VOL*100:.2f}%.
- Structural stop below base; next-open risk <= 4%.

TESTED ENTRY TIMING
- Reversal day while still inside the dry base.
- Close above prior 2-day high.
- Close above prior 3-day high.
- Close above full 10-day base high.

EXPANSION-LINKED VARIANTS
- Prior 60-day pre-base thrust >= 30%.
- At least two >=2x-volume days before the base.
- Dry base volume <= 60% of the preceding expansion's peak 5-day average volume.

MARKET REGIME VARIANTS
- No market gate.
- Composite_Score >= 40.
- Composite_Score >= 50.
The score is known at signal-day close; entry remains next-session open.

CONCENTRATION
- Top-winner contribution and results after removing top 5/10 trades are reported separately.
""", encoding="utf-8")

    print(f"Completed trades across all combinations: {len(trades):,}")
    if not recommended.empty:
        print("\nTOP CANDIDATES")
        show = [c for c in [
            "EntryConfig", "ExitStrategy", "RobustnessScore", "Train_Trades", "OOS_Trades",
            "Train_GeoMeanPerTrade_pct", "OOS_GeoMeanPerTrade_pct", "OOS_ProfitFactor",
            "OOS_MedianTrendMonetisationRatio", "OOS_BigWinnerMedianMonetisation",
        ] if c in recommended.columns]
        print(recommended[show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
