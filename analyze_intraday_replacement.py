from __future__ import annotations

import datetime
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from kiteconnect import KiteConnect

from kite_helper import fetch_with_backoff, get_kite_client


ROOT = Path(__file__).resolve().parent

MASTER_FILE = ROOT / "nse_6yr_historical.parquet"
EOD_AGGREGATE_FILE = ROOT / "historical_breadth_regime_6yr.csv"

LIVE_AGGREGATE_FILE = ROOT / "live_intraday_aggregate.csv"
LIVE_FILE = ROOT / "live_intraday.parquet"
DASHBOARD_FILE = ROOT / "dashboard_data.parquet"
BREADTH_FILE = ROOT / "live_intraday_breadth.csv"
SYNC_FILE = ROOT / "last_sync.txt"
AUDIT_DIR = ROOT / "intraday_audits"

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
NOW_IST = datetime.datetime.now(IST)
TODAY = pd.Timestamp(NOW_IST.date()).normalize()

MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE_BUFFER = datetime.time(15, 35)
CALENDAR_LOOKBACK_DAYS = 15

QUOTE_CHUNK_SIZE = 200
MAX_RETRIES = 5
MIN_INTRADAY_COVERAGE = 0.80

NSE_EQUITY_LIST_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

CRITICAL_SYMBOLS = {
    symbol.strip().upper()
    for symbol in os.environ.get(
        "INTRADAY_CRITICAL_SYMBOLS",
        "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS",
    ).split(",")
    if symbol.strip()
}

REQUIRED_COLUMNS = [
    "Symbol",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def skip(message: str) -> None:
    print(f"ℹ️ {message}")
    raise SystemExit(0)


def normalize_date(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")

    if pd.isna(ts):
        return pd.NaT

    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)

    return ts.normalize()


def valid_date(value) -> bool:
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def clean_symbol(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def atomic_write(
    frame: pd.DataFrame,
    destination: Path,
    csv: bool = False,
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")

    if csv:
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False)

    temporary.replace(destination)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "".join(
            json.dumps(record, default=str) + "
"
            for record in records
        ),
        encoding="utf-8",
    )


def kite_call(
    fetcher,
    label: str,
) -> object | None:
    result = fetch_with_backoff(fetcher)

    if result is None:
        print(
            f"⚠️ {label} failed after shared Kite retry/backoff handling."
        )

    return result


def download_current_nse_eq_symbols() -> set[str]:
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                NSE_EQUITY_LIST_URL,
                headers=NSE_HEADERS,
                timeout=30,
            )

            response.raise_for_status()

            if not response.text.strip():
                raise ValueError("NSE EQUITY_L.csv response is empty.")

            listed = pd.read_csv(StringIO(response.text))

            listed.columns = [
                str(column).strip().upper()
                for column in listed.columns
            ]

            required = {"SYMBOL", "SERIES"}
            missing = required - set(listed.columns)

            if missing:
                raise ValueError(
                    "NSE EQUITY_L.csv missing required columns: "
                    f"{sorted(missing)}"
                )

            listed["SYMBOL"] = listed["SYMBOL"].map(clean_symbol)

            listed["SERIES"] = (
                listed["SERIES"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            symbols = set(
                listed.loc[
                    (listed["SERIES"] == "EQ")
                    & (listed["SYMBOL"] != ""),
                    "SYMBOL",
                ]
            )

            if len(symbols) < 1000:
                raise ValueError(
                    "NSE current EQ universe looks incomplete: "
                    f"{len(symbols)} symbols."
                )

            return symbols

        except Exception as exc:
            last_error = exc

            print(
                "⚠️ NSE EQUITY_L.csv download attempt "
                f"{attempt + 1}/{MAX_RETRIES} failed: {exc}"
            )

            time.sleep(min(2 ** attempt, 16))

    fail(
        "Could not download a reliable NSE EQUITY_L.csv: "
        f"{last_error}"
    )


def confirm_nse_trading_session(kite: KiteConnect) -> None:
    if (
        NOW_IST.time() < MARKET_OPEN
        or NOW_IST.time() > MARKET_CLOSE_BUFFER
    ):
        skip(
            "Outside configured NSE intraday window "
            f"({MARKET_OPEN.strftime('%I:%M %p')}–"
            f"{MARKET_CLOSE_BUFFER.strftime('%I:%M %p')} IST). "
            "Intraday update skipped safely."
        )

    nifty_result = kite_call(
        lambda: kite.quote(["NSE:NIFTY 50"]),
        "NIFTY 50 quote",
    )

    nifty_data = (
        nifty_result.get("NSE:NIFTY 50")
        if nifty_result
        else None
    )

    if not nifty_data:
        fail(
            "Could not retrieve NIFTY 50 quote "
            "to confirm the NSE trading session."
        )

    nifty_token = nifty_data.get("instrument_token")

    if not nifty_token:
        fail("NIFTY 50 instrument token is missing.")

    calendar_start = TODAY - pd.Timedelta(
        days=CALENDAR_LOOKBACK_DAYS
    )

    calendar = kite_call(
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
        skip(
            f"NSE is not trading on {TODAY.date()}. "
            f"Latest confirmed NIFTY market date: "
            f"{market_dates[-1].date()}. "
            "Intraday update skipped safely."
        )

    last_trade_date = normalize_date(
        nifty_data.get("last_trade_time")
    )

    if valid_date(last_trade_date) and last_trade_date < TODAY:
        skip(
            "NIFTY quote is stale "
            f"(last trade date: {last_trade_date.date()}). "
            "Intraday update skipped safely."
        )

    print(
        f"✅ NSE trading session confirmed for {TODAY.date()} "
        f"at {NOW_IST.strftime('%I:%M %p IST')}."
    )


def is_valid_live_ohlcv(row: pd.Series) -> bool:
    try:
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        volume = float(row["Volume"])

        return (
            open_price > 0
            and high_price > 0
            and low_price > 0
            and close_price > 0
            and volume >= 0
            and high_price >= low_price
            and high_price >= max(open_price, close_price)
            and low_price <= min(open_price, close_price)
        )

    except (KeyError, TypeError, ValueError):
        return False


def fetch_live_quotes(
    kite: KiteConnect,
    symbols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    failures = []

    def fetch_chunk(
        chunk: list[str],
        chunk_label: str,
    ) -> None:
        instruments = [
            f"NSE:{symbol}"
            for symbol in chunk
        ]

        response = kite_call(
            lambda: kite.quote(instruments),
            f"live quote {chunk_label}",
        )

        if response is None:
            if len(chunk) == 1:
                failures.append(
                    {
                        "Symbol": chunk[0],
                        "Reason": "quote_request_failed",
                    }
                )
                return

            midpoint = len(chunk) // 2

            fetch_chunk(
                chunk[:midpoint],
                f"{chunk_label}.L",
            )

            fetch_chunk(
                chunk[midpoint:],
                f"{chunk_label}.R",
            )

            return

        returned_symbols = set()

        for key, data in response.items():
            symbol = clean_symbol(
                key.replace("NSE:", "", 1)
            )

            if symbol not in chunk:
                continue

            returned_symbols.add(symbol)

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

        for symbol in sorted(
            set(chunk) - returned_symbols
        ):
            failures.append(
                {
                    "Symbol": symbol,
                    "Reason": "missing_from_quote_response",
                }
            )

    for start in range(0, len(symbols), QUOTE_CHUNK_SIZE):
        chunk = symbols[
            start:start + QUOTE_CHUNK_SIZE
        ]

        fetch_chunk(
            chunk,
            str(start // QUOTE_CHUNK_SIZE + 1),
        )

    live = pd.DataFrame(
        rows,
        columns=REQUIRED_COLUMNS,
    )

    if live.empty:
        fail("Kite returned no live quotes.")

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        live[column] = pd.to_numeric(
            live[column],
            errors="coerce",
        )

    valid_rows = live.apply(
        is_valid_live_ohlcv,
        axis=1,
    )

    invalid_symbols = set(
        live.loc[
            ~valid_rows,
            "Symbol",
        ].map(clean_symbol)
    )

    live = live[valid_rows].copy()

    live["Date"] = pd.to_datetime(
        live["Date"]
    ).dt.normalize()

    live = live.drop_duplicates(
        ["Symbol", "Date"],
        keep="last",
    )

    existing_failure_symbols = {
        record.get("Symbol")
        for record in failures
    }

    for symbol in sorted(
        invalid_symbols - existing_failure_symbols
    ):
        failures.append(
            {
                "Symbol": symbol,
                "Reason": "invalid_live_ohlcv",
            }
        )

    return live, failures


def prepare_stock_data(
    master: pd.DataFrame,
    live: pd.DataFrame,
) -> pd.DataFrame:
    df = master.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    ).dt.normalize()

    df["Symbol"] = df["Symbol"].map(clean_symbol)

    df = df.dropna(subset=["Date"])
    df = df[df["Symbol"] != ""]
    df = df[df["Date"] != TODAY]

    df = pd.concat(
        [df, live],
        ignore_index=True,
        sort=False,
    )

    df = (
        df.sort_values(["Symbol", "Date"])
        .drop_duplicates(
            ["Symbol", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    group = df.groupby("Symbol", group_keys=False)

    df["History_Days"] = group.cumcount() + 1
    df["Prior_History_Days"] = df["History_Days"] - 1

    df["Daily_Turnover"] = (
        df["Close"] * df["Volume"]
    )

    df["Prior_Turnover_20D_Avg"] = group[
        "Daily_Turnover"
    ].transform(
        lambda x: x.shift(1).rolling(
            20,
            min_periods=1,
        ).mean()
    )

    mature_valid = (
        (df["Prior_History_Days"] >= 20)
        & (
            df["Prior_Turnover_20D_Avg"]
            >= 50_000_000
        )
    )

    new_valid = (
        (df["Prior_History_Days"] >= 1)
        & (df["Prior_History_Days"] < 20)
    )

    df["Active_Universe"] = (
        (mature_valid | new_valid)
        & (df["Volume"] > 0)
    )

    df["Turnover_45d_Avg"] = group[
        "Daily_Turnover"
    ].transform(
        lambda x: x.shift(1).rolling(
            45,
            min_periods=1,
        ).mean()
    )

    df["Cap_Rank"] = df.groupby("Date")[
        "Turnover_45d_Avg"
    ].rank(
        ascending=False,
        method="min",
    )

    df["Liquidity_Category"] = np.select(
        [
            df["Cap_Rank"] <= 100,
            (
                (df["Cap_Rank"] > 100)
                & (df["Cap_Rank"] <= 250)
            ),
            (
                (df["Cap_Rank"] > 250)
                & (df["Cap_Rank"] <= 500)
            ),
        ],
        [
            "Top 100 Liq",
            "Mid 150 Liq",
            "Lower 250 Liq",
        ],
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
        lambda x: x.shift(1).rolling(
            252,
            min_periods=1,
        ).max()
    )

    df["Rolling_52W_Low"] = group["Low"].transform(
        lambda x: x.shift(1).rolling(
            252,
            min_periods=1,
        ).min()
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
    ).max(axis=1).fillna(
        df["High"] - df["Low"]
    )

    df["TR"] = true_range

    df["ATR_14"] = group["TR"].transform(
        lambda x: x.ewm(
            alpha=1 / 14,
            min_periods=1,
            adjust=False,
        ).mean()
    )

    df["Vol_20D_Avg"] = group["Volume"].transform(
        lambda x: x.shift(1).rolling(
            20,
            min_periods=1,
        ).mean()
    )

    df["Max_20D_Prior"] = group["High"].transform(
        lambda x: x.shift(1).rolling(
            20,
            min_periods=1,
        ).max()
    )

    df["Listing_Day_High"] = group["High"].transform(
        "first"
    )

    df["VCP_Tightness"] = (
        df["ATR_14"] / df["Close"]
    ) < 0.04

    # This only identifies a possible breakout that occurred today.
    # It does NOT alter today's completed T+3 score.
    df["Volume_Surge"] = (
        df["Volume"]
        > (df["Vol_20D_Avg"] * 1.5)
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

    # Raw intraday ₹ turnover in gaining versus declining active stocks.
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
    part = df[
        df["Liquidity_Category"] == name
    ]

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
            (
                result[f"Above_{period}"]
                / result[f"Valid_{period}"]
                * 100
            ),
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


def intraday_volume_points(
    volume_ratio: pd.Series,
) -> pd.Series:
    """
    Intraday C4 volume score only.

    Uses no smoothing, no historical percentile, and no comparison with
    prior-day/full-day volume. It scores today's current buying turnover
    divided by today's current selling turnover.
    """
    return pd.Series(
        np.select(
            [
                volume_ratio < 0.80,
                (volume_ratio >= 0.80)
                & (volume_ratio < 1.00),
                (volume_ratio >= 1.00)
                & (volume_ratio < 1.20),
                (volume_ratio >= 1.20)
                & (volume_ratio < 1.50),
                (volume_ratio >= 1.50)
                & (volume_ratio < 2.00),
                volume_ratio >= 2.00,
            ],
            [
                0,
                2,
                4,
                6,
                8,
                10,
            ],
            default=0,
        ),
        index=volume_ratio.index,
        dtype=float,
    )


def calculate_aggregate(
    stock: pd.DataFrame,
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
            (
                result[f"Above_{period}_EMA"]
                / result[f"Valid_{period}"]
                * 100
            ),
            np.nan,
        )

    result["Rolling_3D_Up_4"] = (
        result["Up_4_Count"]
        .rolling(3, min_periods=1)
        .sum()
    )

    result["Rolling_3D_Down_4"] = (
        result["Down_4_Count"]
        .rolling(3, min_periods=1)
        .sum()
    )

    result["Net_52W_High_Low"] = (
        result["New_52W_Highs"]
        - result["New_52W_Lows"]
    )

    adv = result["Advances"].astype(float)
    dec = result["Declines"].astype(float)
    uv = result["Total_Up_Volume"].astype(float)
    dv = result["Total_Down_Volume"].astype(float)

    # Simple current-session buying turnover / selling turnover ratio.
    result["Volume_Ratio"] = np.where(
        dv > 0,
        uv / dv,
        np.nan,
    )

    result["AD_Spread"] = adv - dec

    result["MCO"] = (
        result["AD_Spread"]
        .ewm(span=19, adjust=False)
        .mean()
        - result["AD_Spread"]
        .ewm(span=39, adjust=False)
        .mean()
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

    # C1: Core Breadth (0 to 25)
    p_blend = (
        0.65 * result["Pct_Above_20_EMA"].fillna(0)
        + 0.35 * result["Pct_Above_50_EMA"].fillna(0)
    )

    c1 = (p_blend / 100) * 25

    # C2: Breakout Health (0 to 25)
    # Raw daily T3 columns remain available for dashboard display.
    # Composite score uses the trailing five-session total only.
    result["T3_Breakouts_5D"] = (
        result["T3_Breakouts"]
        .astype(float)
        .rolling(5, min_periods=1)
        .sum()
    )

    result["T3_Wins_5D"] = (
        result["T3_Wins"]
        .astype(float)
        .rolling(5, min_periods=1)
        .sum()
    )

    t3_breakouts_5d = result["T3_Breakouts_5D"]
    t3_wins_5d = result["T3_Wins_5D"]

    smoothed = (
        (t3_wins_5d + 5)
        / (t3_breakouts_5d + 10)
    )

    c2 = (
        np.clip(
            (smoothed - 0.45) / 0.15,
            0,
            1,
        )
        * 25
        * np.clip(
            t3_breakouts_5d / 10,
            0,
            1,
        )
    )

    # C3: Momentum Thrust (0 to 20)
    net_4d = (
        result["Rolling_3D_Up_4"]
        - result["Rolling_3D_Down_4"]
    )

    net_1m = (
        result["Up_25_1M_Count"]
        - result["Down_25_1M_Count"]
    )

    rank_4d = (
        net_4d
        .rolling(126, min_periods=1)
        .rank(pct=True)
    )

    rank_1m = (
        net_1m
        .rolling(126, min_periods=1)
        .rank(pct=True)
    )

    c3 = (
        rank_4d * 10
        + rank_1m * 10
    )

    # C4: Intraday Volume and Net New Highs/Lows (0 to 20)
    # Volume points are raw fixed bands, with no historical smoothing/rank.
    result["Intraday_Volume_Points"] = intraday_volume_points(
        result["Volume_Ratio"]
    )

    rank_hl = (
        result["Net_52W_High_Low"]
        .rolling(126, min_periods=1)
        .rank(pct=True)
    )

    c4 = (
        result["Intraday_Volume_Points"]
        + np.where(
            result["Net_52W_High_Low"] > 0,
            rank_hl * 10,
            0,
        )
    )

    # C5: Structural Trend (0, 5, or 10)
    p200 = result["Pct_Above_200_EMA"].fillna(0)

    c5 = np.where(
        (p200 > 50)
        & (p200.diff(20).fillna(0) > 0),
        10,
        np.where(
            (p200 <= 50)
            & (p200.diff(20).fillna(0) < 0),
            0,
            5,
        ),
    )

    # C6: Divergence Penalty (-15 to 0)
    hunting = (
        result["Small_Pct_50_EMA"].fillna(0)
        + result["Micro_Pct_50_EMA"].fillna(0)
    ) / 2

    c6 = -np.clip(
        (
            result["Large_Pct_50_EMA"].fillna(0)
            - hunting
            - 20
        )
        * 0.75,
        0,
        15,
    )

    # C7: Washout Recovery Bonus (0 or +15)
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
        columns=[
            "Valid_20",
            "Valid_50",
            "Valid_200",
        ]
    )


def merge_live_today_with_eod_history(
    intraday_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    intraday_aggregate = intraday_aggregate.copy()

    intraday_aggregate["Date"] = pd.to_datetime(
        intraday_aggregate["Date"],
        errors="coerce",
    ).dt.normalize()

    intraday_aggregate = intraday_aggregate.dropna(
        subset=["Date"]
    )

    intraday_today = intraday_aggregate[
        intraday_aggregate["Date"] == TODAY
    ].copy()

    if intraday_today.empty:
        fail("No current-day intraday aggregate row was generated.")

    if not EOD_AGGREGATE_FILE.exists():
        print(
            "⚠️ Permanent EOD aggregate file is unavailable. "
            "Writing intraday aggregate with calculated history."
        )

        return (
            intraday_aggregate
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

    try:
        eod_aggregate = pd.read_csv(EOD_AGGREGATE_FILE)

    except Exception as exc:
        print(
            "⚠️ Could not read permanent EOD aggregate file: "
            f"{type(exc).__name__}: {exc}. "
            "Writing intraday aggregate with calculated history."
        )

        return (
            intraday_aggregate
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

    if "Date" not in eod_aggregate.columns:
        print(
            "⚠️ Permanent EOD aggregate file has no Date column. "
            "Writing intraday aggregate with calculated history."
        )

        return (
            intraday_aggregate
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

    eod_aggregate["Date"] = pd.to_datetime(
        eod_aggregate["Date"],
        errors="coerce",
    ).dt.normalize()

    eod_aggregate = eod_aggregate.dropna(
        subset=["Date"]
    )

    eod_history = eod_aggregate[
        eod_aggregate["Date"] < TODAY
    ].copy()

    live_aggregate = pd.concat(
        [eod_history, intraday_today],
        ignore_index=True,
        sort=False,
    )

    return (
        live_aggregate
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    if not MASTER_FILE.exists():
        fail(f"Missing {MASTER_FILE.name}")

    historical = pd.read_parquet(MASTER_FILE)

    missing_columns = set(REQUIRED_COLUMNS).difference(
        historical.columns
    )

    if missing_columns:
        fail(
            "Master data missing columns: "
            f"{sorted(missing_columns)}"
        )

    historical["Symbol"] = historical["Symbol"].map(
        clean_symbol
    )

    historical["Date"] = pd.to_datetime(
        historical["Date"],
        errors="coerce",
    ).dt.normalize()

    historical = historical.dropna(subset=["Date"])

    historical = historical[
        historical["Symbol"] != ""
    ].copy()

    historical_symbols = set(historical["Symbol"])

    if not historical_symbols:
        fail("No symbols found in the master parquet.")

    kite = get_kite_client()

    confirm_nse_trading_session(kite)

    nse_eq_symbols = download_current_nse_eq_symbols()

    required_symbols = (
        nse_eq_symbols & historical_symbols
    )

    current_eq_without_history = (
        nse_eq_symbols - historical_symbols
    )

    historical_not_current_eq = (
        historical_symbols - nse_eq_symbols
    )

    if len(required_symbols) < 1000:
        fail(
            "Intraday eligible universe is unexpectedly small "
            "after intersecting NSE current EQ symbols with "
            "the historical parquet."
        )

    critical_missing_from_universe = (
        CRITICAL_SYMBOLS - required_symbols
    )

    if critical_missing_from_universe:
        fail(
            "Critical symbols are not in the intraday eligible universe: "
            f"{sorted(critical_missing_from_universe)}"
        )

    print(
        f"NSE current mainboard EQ symbols: "
        f"{len(nse_eq_symbols)}"
    )

    print(
        f"Intraday eligible symbols with usable history: "
        f"{len(required_symbols)}"
    )

    print(
        "Current EQ symbols excluded because history is unavailable: "
        f"{len(current_eq_without_history)}"
    )

    print(
        "Historical symbols excluded because they are no longer "
        f"current EQ: {len(historical_not_current_eq)}"
    )

    live, quote_failures = fetch_live_quotes(
        kite,
        sorted(required_symbols),
    )

    live_symbols = set(
        live["Symbol"].map(clean_symbol)
    )

    missing_live_symbols = (
        required_symbols - live_symbols
    )

    critical_missing_live = (
        CRITICAL_SYMBOLS & missing_live_symbols
    )

    coverage = (
        len(live_symbols) / len(required_symbols)
    )

    audit_stamp = NOW_IST.strftime("%Y%m%d_%H%M%S")

    audit_records = [
        {
            "Symbol": symbol,
            "Date": TODAY,
            "Reason": (
                "current_nse_eq_without_historical_data"
            ),
        }
        for symbol in sorted(current_eq_without_history)
    ]

    audit_records.extend(
        {
            "Symbol": symbol,
            "Date": TODAY,
            "Reason": (
                "eligible_current_eq_missing_or_invalid_live_quote"
            ),
        }
        for symbol in sorted(missing_live_symbols)
    )

    audit_records.extend(
        {
            "Date": TODAY,
            "Reason": "quote_fetch_detail",
            **failure,
        }
        for failure in quote_failures
    )

    if audit_records:
        write_jsonl(
            AUDIT_DIR / (
                f"intraday_coverage_{audit_stamp}.jsonl"
            ),
            audit_records,
        )

    print(
        f"Live valid quote coverage: "
        f"{len(live_symbols)}/{len(required_symbols)} "
        f"({coverage:.2%})."
    )

    if critical_missing_live:
        fail(
            "Critical intraday symbols missing or invalid: "
            f"{sorted(critical_missing_live)}."
        )

    if coverage < MIN_INTRADAY_COVERAGE:
        fail(
            f"Intraday coverage below required "
            f"{MIN_INTRADAY_COVERAGE:.0%}: "
            f"{len(live_symbols)}/{len(required_symbols)} "
            f"({coverage:.2%})."
        )

    stock_data = prepare_stock_data(
        historical,
        live,
    )

    calculated_aggregate = calculate_aggregate(
        stock_data
    )

    live_aggregate = merge_live_today_with_eod_history(
        calculated_aggregate
    )

    today_row = live_aggregate[
        live_aggregate["Date"] == TODAY
    ]

    if today_row.empty:
        fail("No current-day live aggregate row was generated.")

    atomic_write(
        live,
        LIVE_FILE,
    )

    atomic_write(
        stock_data,
        DASHBOARD_FILE,
    )

    atomic_write(
        live_aggregate,
        LIVE_AGGREGATE_FILE,
        csv=True,
    )

    atomic_write(
        today_row[
            [
                "Date",
                "Advances",
                "Declines",
                "Unchanged",
                "Total_Universe",
            ]
        ].assign(
            Time=NOW_IST.strftime("%H:%M")
        ),
        BREADTH_FILE,
        csv=True,
    )

    SYNC_FILE.write_text(
        (
            "LIVE INTRADAY | "
            f"{TODAY.strftime('%d %B %Y')} | "
            f"{NOW_IST.strftime('%I:%M %p IST')} | "
            f"Coverage {coverage:.2%}
"
        ),
        encoding="utf-8",
    )

    current = today_row.iloc[0]

    write_json(
        AUDIT_DIR / (
            f"intraday_run_{audit_stamp}.json"
        ),
        {
            "date": str(TODAY.date()),
            "time_ist": NOW_IST.strftime("%I:%M %p IST"),
            "status": "LIVE_INTRADAY",
            "nse_current_eq_symbols": len(nse_eq_symbols),
            "eligible_symbols_with_history": len(required_symbols),
            "current_eq_without_history": len(
                current_eq_without_history
            ),
            "historical_not_current_eq": len(
                historical_not_current_eq
            ),
            "valid_live_quotes": len(live_symbols),
            "missing_or_invalid_live_quotes": len(
                missing_live_symbols
            ),
            "coverage": round(coverage, 6),
            "minimum_coverage": MIN_INTRADAY_COVERAGE,
            "critical_missing": sorted(
                critical_missing_live
            ),
            "quote_failure_records": len(quote_failures),
            "live_aggregate_file": LIVE_AGGREGATE_FILE.name,
            "composite_score": int(
                current["Composite_Score"]
            ),
            "volume_ratio": (
                float(current["Volume_Ratio"])
                if pd.notna(current["Volume_Ratio"])
                else None
            ),
            "intraday_volume_points": int(
                current["Intraday_Volume_Points"]
            ),
            "daily_t3_breakouts": int(
                current["T3_Breakouts"]
            ),
            "daily_t3_wins": int(
                current["T3_Wins"]
            ),
            "t3_breakouts_5d_for_score": int(
                current["T3_Breakouts_5D"]
            ),
            "t3_wins_5d_for_score": int(
                current["T3_Wins_5D"]
            ),
        },
    )

    print(f"✅ Live aggregate generated: {TODAY.date()}")
    print(
        f"✅ Composite Score: "
        f"{int(current['Composite_Score'])}"
    )
    print(
        f"✅ Volume Ratio: "
        f"{current['Volume_Ratio']:.2f}"
        if pd.notna(current["Volume_Ratio"])
        else "✅ Volume Ratio: N/A"
    )
    print(
        f"✅ Intraday Volume Points: "
        f"{int(current['Intraday_Volume_Points'])}/10"
    )
    print(
        f"✅ Daily T+3: "
        f"{int(current['T3_Wins'])}/"
        f"{int(current['T3_Breakouts'])}"
    )
    print(
        f"✅ Five-Day T+3 Score Cohort: "
        f"{int(current['T3_Wins_5D'])}/"
        f"{int(current['T3_Breakouts_5D'])}"
    )
    print(f"✅ MCO: {current['MCO']}")
    print(f"✅ TRIN: {current['TRIN']}")


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
        raise SystemExit(1)
