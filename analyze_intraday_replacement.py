from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect


ROOT = Path(__file__).resolve().parent
MASTER_FILE = ROOT / "nse_6yr_historical.parquet"
AGGREGATE_FILE = ROOT / "historical_breadth_regime_6yr.csv"
TRAILING_FILE = ROOT / "trailing_cache.parquet"
LIVE_FILE = ROOT / "live_intraday.parquet"
DASHBOARD_FILE = ROOT / "dashboard_data.parquet"
BREADTH_FILE = ROOT / "live_intraday_breadth.csv"
SYNC_FILE = ROOT / "last_sync.txt"

API_KEY = os.environ.get("KITE_API_KEY", "").strip()
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "").strip()

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
NOW_IST = datetime.datetime.now(IST)
TODAY = pd.Timestamp(NOW_IST.date()).normalize()

MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE_BUFFER = datetime.time(15, 35)
CALENDAR_LOOKBACK_DAYS = 15

QUOTE_CHUNK_SIZE = 200
MAX_RETRIES = 5
QUOTE_SLEEP_SECONDS = 1.1

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)


def skip(message: str) -> None:
    """Exit successfully without writing any intraday or dashboard files."""
    print(f"ℹ️ {message}")
    sys.exit(0)


def normalize_date(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT

    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)

    return ts.normalize()


def valid_date(value) -> bool:
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def atomic_write(frame: pd.DataFrame, destination: Path, csv: bool = False) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")

    if csv:
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)

    temporary.replace(destination)


def fetch_retry(fetcher, label: str):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            result = fetcher()
            if result is not None:
                return result
        except Exception as exc:
            last_error = exc
            print(
                f"⚠️ {label} attempt {attempt + 1}/{MAX_RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )

        time.sleep(min(2 ** attempt, 8))

    return None


def confirm_nse_trading_session(kite: KiteConnect) -> None:
    """
    Confirm that the current IST date is an actual NSE trading day before
    fetching quotes or writing any live dashboard data.

    NIFTY 50 daily candles are used as the authoritative exchange calendar.
    This safely skips ordinary weekends and NSE weekday holidays while allowing
    valid special sessions if NIFTY has a candle for the current date.
    """
    if NOW_IST.time() < MARKET_OPEN or NOW_IST.time() > MARKET_CLOSE_BUFFER:
        skip(
            "Outside configured NSE intraday window "
            f"({MARKET_OPEN.strftime('%I:%M %p')}–"
            f"{MARKET_CLOSE_BUFFER.strftime('%I:%M %p')} IST). "
            "Intraday update skipped safely."
        )

    nifty_result = fetch_retry(
        lambda: kite.quote(["NSE:NIFTY 50"]),
        "NIFTY 50 quote",
    )

    nifty_data = nifty_result.get("NSE:NIFTY 50") if nifty_result else None

    if not nifty_data:
        fail(
            "Could not retrieve NIFTY 50 quote to confirm the NSE trading session."
        )

    nifty_token = nifty_data.get("instrument_token")

    if not nifty_token:
        fail("NIFTY 50 instrument token is missing.")

    calendar_start = TODAY - pd.Timedelta(days=CALENDAR_LOOKBACK_DAYS)

    calendar = fetch_retry(
        lambda: kite.historical_data(
            nifty_token,
            calendar_start.strftime("%Y-%m-%d"),
            TODAY.strftime("%Y-%m-%d"),
            "day",
        ),
        "NIFTY 50 trading calendar",
    )

    if not calendar:
        fail(
            "Could not retrieve NIFTY trading calendar. "
            "Refusing to write intraday data."
        )

    market_dates = sorted(
        {
            normalize_date(candle.get("date"))
            for candle in calendar
            if valid_date(normalize_date(candle.get("date")))
        }
    )

    if not market_dates:
        fail(
            "NIFTY trading calendar returned no valid dates. "
            "Refusing to write intraday data."
        )

    if TODAY not in set(market_dates):
        latest_market_date = market_dates[-1]

        skip(
            f"NSE is not trading on {TODAY.date()}. "
            f"Latest confirmed NIFTY market date: {latest_market_date.date()}. "
            "Intraday update skipped safely."
        )

    last_trade_date = normalize_date(nifty_data.get("last_trade_time"))

    if valid_date(last_trade_date) and last_trade_date < TODAY:
        skip(
            f"NIFTY quote is stale (last trade date: {last_trade_date.date()}). "
            "Intraday update skipped safely."
        )

    print(
        f"✅ NSE trading session confirmed for {TODAY.date()} "
        f"at {NOW_IST.strftime('%I:%M %p IST')}."
    )


def fetch_live_quotes(kite: KiteConnect, symbols: list[str]) -> pd.DataFrame:
    rows = []
    instruments = [f"NSE:{symbol}" for symbol in symbols]

    for start in range(0, len(instruments), QUOTE_CHUNK_SIZE):
        chunk = instruments[start:start + QUOTE_CHUNK_SIZE]
        response = None

        for attempt in range(MAX_RETRIES):
            try:
                response = kite.quote(chunk)
                if response:
                    break
            except Exception as exc:
                print(
                    f"⚠️ Quote attempt failed for chunk "
                    f"{start // QUOTE_CHUNK_SIZE + 1}, "
                    f"attempt {attempt + 1}/{MAX_RETRIES}: "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(min(2 ** attempt, 8))

        if not response:
            fail(f"No quote response for chunk {start // QUOTE_CHUNK_SIZE + 1}")

        for key, data in response.items():
            symbol = key.replace("NSE:", "", 1)
            ohlc = data.get("ohlc") or {}

            rows.append(
                {
                    "Symbol": symbol,
                    "Date": TODAY,
                    "Open": ohlc.get("open"),
                    "High": ohlc.get("high"),
                    "Low": ohlc.get("low"),
                    "Close": data.get("last_price"),
                    "Volume": data.get("volume", 0),
                }
            )

        time.sleep(QUOTE_SLEEP_SECONDS)

    live = pd.DataFrame(rows)

    if live.empty:
        fail("Kite returned no live quotes.")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        live[column] = pd.to_numeric(live[column], errors="coerce")

    live = live.dropna(subset=["Symbol", "Close"])
    live["Date"] = pd.to_datetime(live["Date"]).dt.normalize()

    return live.drop_duplicates(["Symbol", "Date"], keep="last")


def prepare_stock_data(master: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df.dropna(subset=["Date", "Symbol"])

    df = df[df["Date"] != TODAY]
    df = pd.concat([df, live], ignore_index=True, sort=False)

    df = (
        df.sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
        .reset_index(drop=True)
    )

    group = df.groupby("Symbol", group_keys=False)

    df["History_Days"] = group.cumcount() + 1
    df["Prior_History_Days"] = df["History_Days"] - 1

    df["Daily_Turnover"] = df["Close"] * df["Volume"]

    df["Prior_Turnover_20D_Avg"] = group["Daily_Turnover"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    df["Active_Universe"] = (
        (
            (df["Prior_History_Days"] >= 20)
            & (df["Prior_Turnover_20D_Avg"] >= 50_000_000)
        )
        | (
            (df["Prior_History_Days"] >= 1)
            & (df["Prior_History_Days"] < 20)
        )
    ) & (df["Volume"] > 0)

    df["Turnover_45d_Avg"] = group["Daily_Turnover"].transform(
        lambda x: x.shift(1).rolling(45, min_periods=1).mean()
    )

    df["Cap_Rank"] = df.groupby("Date")["Turnover_45d_Avg"].rank(
        ascending=False,
        method="min",
    )

    df["Liquidity_Category"] = np.select(
        [
            df["Cap_Rank"] <= 100,
            (df["Cap_Rank"] > 100) & (df["Cap_Rank"] <= 250),
            (df["Cap_Rank"] > 250) & (df["Cap_Rank"] <= 500),
        ],
        ["Top 100 Liq", "Mid 150 Liq", "Lower 250 Liq"],
        default="Micro Liq",
    )

    for period in [20, 50, 200]:
        df[f"EMA_{period}"] = group["Close"].transform(
            lambda x, p=period: x.ewm(
                span=p,
                adjust=False,
                min_periods=1,
            ).mean()
        )

        df[f"Valid_{period}_EMA"] = (
            df["Active_Universe"]
            & (df["Prior_History_Days"] >= period)
        )

        df[f"Above_{period}_EMA"] = (
            df[f"Valid_{period}_EMA"]
            & (df["Close"] > df[f"EMA_{period}"])
        )

    df["Prev_Close"] = group["Close"].shift(1)
    df["Has_Prior_Close"] = df["Prev_Close"].notna()

    df["Daily_Pct"] = group["Close"].pct_change() * 100
    df["Pct_1M"] = group["Close"].pct_change(21) * 100

    df["Gainer"] = (
        df["Active_Universe"]
        & df["Has_Prior_Close"]
        & (df["Daily_Pct"] > 0)
    )

    df["Loser"] = (
        df["Active_Universe"]
        & df["Has_Prior_Close"]
        & (df["Daily_Pct"] < 0)
    )

    df["Rolling_52W_High"] = group["High"].transform(
        lambda x: x.shift(1).rolling(252, min_periods=1).max()
    )

    df["Rolling_52W_Low"] = group["Low"].transform(
        lambda x: x.shift(1).rolling(252, min_periods=1).min()
    )

    df["Prior_ATH"] = group["High"].transform(
        lambda x: x.shift(1).cummax()
    )

    df["New_52W_High"] = (
        df["Active_Universe"]
        & (df["Prior_History_Days"] >= 252)
        & (df["Close"] >= df["Rolling_52W_High"])
    )

    df["New_52W_Low"] = (
        df["Active_Universe"]
        & (df["Prior_History_Days"] >= 252)
        & (df["Close"] <= df["Rolling_52W_Low"])
    )

    df["IPO_New_High"] = (
        df["Active_Universe"]
        & (df["Prior_History_Days"] >= 1)
        & (df["Prior_History_Days"] < 252)
        & (df["Close"] >= df["Prior_ATH"])
    )

    df["Up_4_Pct"] = (
        df["Active_Universe"]
        & (df["Daily_Pct"] >= 4.0)
    )

    df["Down_4_Pct"] = (
        df["Active_Universe"]
        & (df["Daily_Pct"] <= -4.0)
    )

    df["Up_25_1M"] = (
        df["Active_Universe"]
        & (df["Pct_1M"] >= 25.0)
    )

    df["Down_25_1M"] = (
        df["Active_Universe"]
        & (df["Pct_1M"] <= -25.0)
    )

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1).fillna(df["High"] - df["Low"])

    df["TR"] = true_range

    df["ATR_14"] = group["TR"].transform(
        lambda x: x.ewm(
            alpha=1 / 14,
            min_periods=1,
            adjust=False,
        ).mean()
    )

    df["Vol_20D_Avg"] = group["Volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    df["Max_20D_Prior"] = group["High"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).max()
    )

    df["Listing_Day_High"] = group["High"].transform("first")

    df["VCP_Tightness"] = (df["ATR_14"] / df["Close"]) < 0.04

    df["Volume_Surge"] = (
        df["Volume"] > (df["Vol_20D_Avg"] * 1.5)
    )

    prior_tight = (
        group["VCP_Tightness"]
        .shift(1)
        .astype(float)
        .fillna(0)
        .astype(bool)
    )

    df["Mature_Breakout"] = (
        (df["Prior_History_Days"] >= 20)
        & (df["Close"] > df["Max_20D_Prior"])
    )

    df["IPO_Breakout"] = (
        (df["Prior_History_Days"] >= 1)
        & (df["Prior_History_Days"] < 20)
        & (df["Close"] > df["Listing_Day_High"])
    )

    df["Is_Breakout"] = (
        df["Active_Universe"]
        & (df["Mature_Breakout"] | df["IPO_Breakout"])
        & df["Volume_Surge"]
        & prior_tight
    )

    df["Is_Breakout_3d_ago"] = (
        group["Is_Breakout"]
        .shift(3)
        .astype(float)
        .fillna(0)
        .astype(bool)
    )

    df["Close_3d_ago"] = group["Close"].shift(3)

    df["Follow_Through_Win"] = (
        df["Is_Breakout_3d_ago"]
        & (df["Close"] > df["Close_3d_ago"])
    )

    df["Up_Volume"] = np.where(
        df["Gainer"],
        df["Daily_Turnover"],
        0,
    )

    df["Down_Volume"] = np.where(
        df["Loser"],
        df["Daily_Turnover"],
        0,
    )

    return df


def liquidity_breadth(
    df: pd.DataFrame,
    name: str,
    prefix: str,
) -> pd.DataFrame:
    part = df[df["Liquidity_Category"] == name]

    result = part.groupby("Date").agg(
        Valid_20=("Valid_20_EMA", "sum"),
        Valid_50=("Valid_50_EMA", "sum"),
        Valid_200=("Valid_200_EMA", "sum"),
        Above_20=("Above_20_EMA", "sum"),
        Above_50=("Above_50_EMA", "sum"),
        Above_200=("Above_200_EMA", "sum"),
    ).reset_index()

    for period in [20, 50, 200]:
        result[f"{prefix}_Pct_{period}_EMA"] = np.where(
            result[f"Valid_{period}"] > 0,
            result[f"Above_{period}"]
            / result[f"Valid_{period}"]
            * 100,
            np.nan,
        )

    return result[
        [
            "Date",
            f"{prefix}_Pct_20_EMA",
            f"{prefix}_Pct_50_EMA",
            f"{prefix}_Pct_200_EMA",
        ]
    ]


def calculate_aggregate(
    stock: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    result = stock.groupby("Date").agg(
        Total_Universe=("Active_Universe", "sum"),
        Valid_20=("Valid_20_EMA", "sum"),
        Valid_50=("Valid_50_EMA", "sum"),
        Valid_200=("Valid_200_EMA", "sum"),
        Advances=("Gainer", "sum"),
        Declines=("Loser", "sum"),
        Above_20_EMA=("Above_20_EMA", "sum"),
        Above_50_EMA=("Above_50_EMA", "sum"),
        Above_200_EMA=("Above_200_EMA", "sum"),
        Up_4_Count=("Up_4_Pct", "sum"),
        Down_4_Count=("Down_4_Pct", "sum"),
        Up_25_1M_Count=("Up_25_1M", "sum"),
        Down_25_1M_Count=("Down_25_1M", "sum"),
        New_52W_Highs=("New_52W_High", "sum"),
        New_52W_Lows=("New_52W_Low", "sum"),
        IPO_New_Highs=("IPO_New_High", "sum"),
        Total_Up_Volume=("Up_Volume", "sum"),
        Total_Down_Volume=("Down_Volume", "sum"),
        T3_Breakouts=("Is_Breakout_3d_ago", "sum"),
        T3_Wins=("Follow_Through_Win", "sum"),
    ).reset_index()

    for period in [20, 50, 200]:
        result[f"Pct_Above_{period}_EMA"] = np.where(
            result[f"Valid_{period}"] > 0,
            result[f"Above_{period}_EMA"]
            / result[f"Valid_{period}"]
            * 100,
            np.nan,
        )

    result["Rolling_3D_Up_4"] = result["Up_4_Count"].rolling(3).sum()
    result["Rolling_3D_Down_4"] = result["Down_4_Count"].rolling(3).sum()
    result["Net_52W_High_Low"] = (
        result["New_52W_Highs"]
        - result["New_52W_Lows"]
    )

    adv = result["Advances"].astype(float)
    dec = result["Declines"].astype(float)
    uv = result["Total_Up_Volume"].astype(float)
    dv = result["Total_Down_Volume"].astype(float)

    result["Volume_Ratio"] = np.where(dv > 0, uv / dv, np.nan)
    result["AD_Spread"] = adv - dec

    result["MCO"] = (
        result["AD_Spread"].ewm(span=19, adjust=False).mean()
        - result["AD_Spread"].ewm(span=39, adjust=False).mean()
    )

    result["TRIN"] = np.where(
        (adv > 0)
        & (dec > 0)
        & (uv > 0)
        & (dv > 0),
        (adv / dec) / (uv / dv),
        np.nan,
    )

    for name, prefix in [
        ("Top 100 Liq", "Large"),
        ("Mid 150 Liq", "Mid"),
        ("Lower 250 Liq", "Small"),
        ("Micro Liq", "Micro"),
    ]:
        result = result.merge(
            liquidity_breadth(stock, name, prefix),
            on="Date",
            how="left",
        )

    result = result.sort_values("Date").reset_index(drop=True)

    p_blend = (
        0.65 * result["Pct_Above_20_EMA"].fillna(0)
        + 0.35 * result["Pct_Above_50_EMA"].fillna(0)
    )

    c1 = (p_blend / 100) * 25

    t3_b = result["T3_Breakouts"].astype(float)
    t3_w = result["T3_Wins"].astype(float)

    smoothed = (t3_w + 5) / (t3_b + 10)

    c2 = (
        np.clip((smoothed - 0.45) / 0.15, 0, 1)
        * 25
        * np.clip(t3_b / 10, 0, 1)
    )

    net_4d = (
        result["Rolling_3D_Up_4"]
        - result["Rolling_3D_Down_4"]
    )

    net_1m = (
        result["Up_25_1M_Count"]
        - result["Down_25_1M_Count"]
    )

    rank_4d = net_4d.rolling(126, min_periods=1).rank(pct=True)
    rank_1m = net_1m.rolling(126, min_periods=1).rank(pct=True)

    c3 = rank_4d * 10 + rank_1m * 10

    rank_vol = (
        result["Volume_Ratio"]
        .fillna(1)
        .rolling(126, min_periods=1)
        .rank(pct=True)
    )

    rank_hl = (
        result["Net_52W_High_Low"]
        .rolling(126, min_periods=1)
        .rank(pct=True)
    )

    c4 = (
        np.where(
            result["Volume_Ratio"].fillna(0) > 1,
            rank_vol * 10,
            0,
        )
        + np.where(
            result["Net_52W_High_Low"] > 0,
            rank_hl * 10,
            0,
        )
    )

    p200 = result["Pct_Above_200_EMA"].fillna(0)

    c5 = np.where(
        (p200 > 50) & (p200.diff(20).fillna(0) > 0),
        10,
        np.where(
            (p200 <= 50) & (p200.diff(20).fillna(0) < 0),
            0,
            5,
        ),
    )

    hunting = (
        result["Small_Pct_50_EMA"].fillna(0)
        + result["Micro_Pct_50_EMA"].fillna(0)
    ) / 2

    c6 = -np.clip(
        (
            result["Large_Pct_50_EMA"].fillna(0)
            - hunting
            - 20
        ) * 0.75,
        0,
        15,
    )

    c7 = np.where(
        (
            result["Pct_Above_20_EMA"]
            .rolling(20, min_periods=1)
            .min()
            <= 10
        )
        & (p_blend >= 50),
        15,
        0,
    )

    result["Composite_Score"] = (
        pd.Series(c1).fillna(0)
        + pd.Series(c2).fillna(0)
        + pd.Series(c3).fillna(0)
        + pd.Series(c4).fillna(0)
        + pd.Series(c5).fillna(0)
        + pd.Series(c6).fillna(0)
        + pd.Series(c7).fillna(0)
    ).clip(0, 100).round().astype(int)

    result["Unchanged"] = (
        result["Total_Universe"]
        - result["Advances"]
        - result["Declines"]
    )

    return result.drop(
        columns=["Valid_20", "Valid_50", "Valid_200"]
    )


def main() -> None:
    if not MASTER_FILE.exists():
        fail(f"Missing {MASTER_FILE.name}")

    if not API_KEY or not ACCESS_TOKEN:
        fail("KITE_API_KEY or KITE_ACCESS_TOKEN is missing.")

    historical = pd.read_parquet(MASTER_FILE)

    required = {
        "Symbol",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required.difference(historical.columns)

    if missing:
        fail(f"Master data missing columns: {sorted(missing)}")

    symbols = sorted(
        historical["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if not symbols:
        fail("No symbols found in the master parquet.")

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    # Must happen before fetch_live_quotes() and before any write operation.
    confirm_nse_trading_session(kite)

    live = fetch_live_quotes(kite, symbols)

    stock_data = prepare_stock_data(historical, live)

    previous = (
        pd.read_csv(AGGREGATE_FILE)
        if AGGREGATE_FILE.exists()
        else pd.DataFrame()
    )

    aggregate = calculate_aggregate(stock_data, previous)

    today_row = aggregate[aggregate["Date"] == TODAY]

    if today_row.empty:
        fail("No current-day aggregate row was generated.")

    atomic_write(live, LIVE_FILE)

    atomic_write(stock_data, DASHBOARD_FILE)

    atomic_write(
        stock_data[
            stock_data["Date"] >= TODAY - pd.Timedelta(days=450)
        ],
        TRAILING_FILE,
    )

    atomic_write(aggregate, AGGREGATE_FILE, csv=True)

    atomic_write(
        today_row[
            [
                "Date",
                "Advances",
                "Declines",
                "Unchanged",
                "Total_Universe",
            ]
        ].assign(Time=NOW_IST.strftime("%H:%M")),
        BREADTH_FILE,
        csv=True,
    )

    SYNC_FILE.write_text(
        f"Today, {NOW_IST.strftime('%I:%M %p')} IST\n",
        encoding="utf-8",
    )

    print(f"✅ Current aggregate generated: {TODAY.date()}")
    print(
        f"✅ Composite Score: "
        f"{int(today_row.iloc[0]['Composite_Score'])}"
    )
    print(f"✅ MCO: {today_row.iloc[0]['MCO']}")
    print(f"✅ TRIN: {today_row.iloc[0]['TRIN']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"❌ Live analytics failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
