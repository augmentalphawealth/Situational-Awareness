"""Runner for Active Leader Swing Backtest v2.

This wrapper patches setup construction so contraction entries never try to access a
nonexistent PRIOR_HIGH_0 column. It also validates breakout trigger windows before use.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import swing_trend_backtest_v2 as core


def build_setups_fixed(df: pd.DataFrame) -> pd.DataFrame:
    common = (
        df["ActiveLeader"]
        & (df["ADV20_PRIOR"] >= core.MIN_TURNOVER_20D)
        & df["NextOpen"].notna()
        & df["BASE_LOW_10"].notna()
        & df["BASE_HIGH_10"].notna()
        & df["ATR_PCT_5"].notna()
        & df["VOL_DRY_10"].notna()
        & df["VOL_CV_10"].notna()
        & df["PRICE_VOL_10"].notna()
    )
    rows: List[pd.DataFrame] = []

    available_breakout_windows = {3, 5, 10}

    for cfg in core.ENTRY_CONFIGS:
        stop = df["BASE_LOW_10"] * 0.995
        entry = df["NextOpen"]
        risk = (entry - stop) / entry

        # Volume dry-up remains a hard part of every reusable setup.
        contraction = (
            (df["BASE_RANGE_10"] <= cfg.max_range)
            & (df["ATR_PCT_5"] <= cfg.max_atr_pct)
            & (df["VOL_DRY_10"] <= cfg.max_vol_dry_ratio)
            & (df["PRICE_VOL_10"] <= cfg.max_price_vol)
            & (df["VOL_CV_10"] <= cfg.max_volume_cv)
        )

        if cfg.trigger == "contraction":
            trigger = df["Close"] <= df["BASE_LOW_10"] * (1.0 + cfg.near_support_pct)
            breakout_level = df["BASE_HIGH_10"]
        elif cfg.trigger == "breakout":
            if cfg.trigger_bars not in available_breakout_windows:
                raise ValueError(
                    f"Unsupported breakout trigger window {cfg.trigger_bars} "
                    f"for entry config {cfg.name}. Available: {sorted(available_breakout_windows)}"
                )
            breakout_col = f"PRIOR_HIGH_{cfg.trigger_bars}"
            trigger = (
                (df["Close"] > df[breakout_col])
                & (df["DAY_VOL_RATIO"] >= cfg.breakout_vol_min)
            )
            breakout_level = df[breakout_col]
        else:
            raise ValueError(f"Unknown trigger type {cfg.trigger!r} for {cfg.name}")

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
        s["BreakoutLevel"] = breakout_level.loc[mask].to_numpy()
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
    return out.sort_values(
        ["EntryConfig", "Symbol", "EntryDate", "SetupScore"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


if __name__ == "__main__":
    core.build_setups = build_setups_fixed
    core.main()
