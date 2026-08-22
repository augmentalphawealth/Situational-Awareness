import datetime
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect

IST = ZoneInfo("Asia/Kolkata")
MORNING_RUN_TIME = datetime.time(9, 30)
EOD_RUN_TIME = datetime.time(17, 0)
QUOTE_CHUNK_SIZE = 200
MAX_RETRIES = 5
QUOTE_SLEEP_SECONDS = 1.1
HISTORY_SLEEP_SECONDS = 0.4
CALENDAR_LOOKBACK_DAYS = 15

PARQUET_FILE = Path("nse_6yr_historical.parquet")
TMP_PARQUET = Path("nse_6yr_historical.tmp.parquet")
LOCK_FILE = Path("eod_sync.lock")
BACKUP_DIR = Path("eod_backups")
AUDIT_DIR = Path("eod_audits")

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
HISTORY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

APPROVED_RATIOS = {
    0.1, 0.125, 0.1667, 0.2, 0.25, 0.3333,
    0.5, 0.6667, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0,
}


def die(message, code=1):
    print(message)
    raise SystemExit(code)


def normalize_date(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def valid_date(value):
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def parse_lock(path):
    try:
        values = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    values[key] = value
        return values
    except OSError as exc:
        die(f"Could not read lock file: {exc}")


def acquire_lock(path):
    if path.exists():
        info = parse_lock(path)
        pid_text = info.get("pid")
        host = info.get("host")
        if not pid_text:
            die("Lock file has no valid PID. Inspect/delete it manually.")
        try:
            pid = int(pid_text)
        except ValueError:
            die("Lock file contains an invalid PID. Inspect/delete it manually.")
        if host and host != socket.gethostname():
            die(f"Lock belongs to host {host}; refusing automatic removal.")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            print(f"Removed stale lock for dead PID {pid}.")
        except PermissionError:
            die(f"Cannot verify lock owner PID {pid}; aborting.")
        except OSError as exc:
            die(f"Could not verify lock owner PID {pid}: {exc}")
        else:
            die(f"Another EOD process is active: PID {pid}.")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            now = datetime.datetime.now(IST)
            handle.write(
                f"pid={os.getpid()}\n"
                f"host={socket.gethostname()}\n"
                f"time={now.isoformat()}\n"
            )
    except FileExistsError:
        die("Another process acquired the lock. Aborting.")


def release_lock(path):
    try:
        info = parse_lock(path)
        if (
            info.get("pid") == str(os.getpid())
            and info.get("host") == socket.gethostname()
        ):
            path.unlink(missing_ok=True)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"Lock cleanup failed: {exc}")


def is_valid_ohlcv(row):
    try:
        opn = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        volume = float(row["Volume"])
        return bool(
            opn > 0 and high > 0 and low > 0 and close > 0
            and high >= max(opn, close, low)
            and low <= min(opn, close, high)
            and volume >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_dataframe(df, name):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"{name} missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die(f"{name} contains invalid dates.")
    if df[NUMERIC_COLUMNS].isna().any().any():
        die(f"{name} contains null numeric values.")
    if df.duplicated(["Symbol", "Date"]).any():
        die(f"{name} contains duplicate Symbol/Date keys.")
    if not df.apply(is_valid_ohlcv, axis=1).all():
        die(f"{name} contains invalid OHLCV rows.")


def fetch_retry(fetcher, label, failures):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return fetcher()
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "403" in str(exc):
                time.sleep(2 ** attempt)
            else:
                time.sleep(min(2 ** attempt, 8))
    failures.append({"label": label, "error": str(last_error)})
    return None


def candles_to_df(candles, symbol):
    if not candles:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.DataFrame(candles).rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    if set(HISTORY_COLUMNS) - set(df.columns):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df["Date"] = df["Date"].apply(normalize_date)
    df["Symbol"] = symbol
    return (
        df[REQUIRED_COLUMNS]
        .sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
    )


def get_token(kite, symbol, failures):
    response = fetch_retry(
        lambda: kite.quote([f"NSE:{symbol}"]),
        f"token {symbol}",
        failures,
    )
    if not response:
        return None
    data = response.get(f"NSE:{symbol}")
    return data.get("instrument_token") if data else None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_jsonl(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, default=str) + "\n" for record in records),
        encoding="utf-8",
    )


def main():
    now = datetime.datetime.now(IST)
    today = normalize_date(now)
    is_weekday = now.weekday() < 5
    is_morning_run = is_weekday and now.time() < EOD_RUN_TIME
    fetch_today = is_weekday and now.time() >= EOD_RUN_TIME
    mode = (
        "morning_recovery" if is_morning_run
        else "eod_recovery" if fetch_today
        else "weekend_or_holiday_recovery"
    )

    acquire_lock(LOCK_FILE)
    try:
        api_key = os.environ.get("KITE_API_KEY")
        access_token = os.environ.get("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            die("KITE_API_KEY or KITE_ACCESS_TOKEN is missing.")
        if TMP_PARQUET.exists():
            die(f"Temporary file exists: {TMP_PARQUET}. Inspect it first.")
        if not PARQUET_FILE.exists():
            die(f"Master database not found: {PARQUET_FILE}")

        source_hash = sha256_file(PARQUET_FILE)
        if source_hash is None:
            die("Could not hash source parquet.")
        df_hist = pd.read_parquet(PARQUET_FILE)
        if sha256_file(PARQUET_FILE) != source_hash:
            die("Source parquet changed while being loaded.")

        missing = set(REQUIRED_COLUMNS) - set(df_hist.columns)
        if missing:
            die(f"Existing database missing columns: {sorted(missing)}")
        df_hist["Date"] = df_hist["Date"].apply(normalize_date)
        validate_dataframe(df_hist, "Existing database")

        db_dates = set(df_hist["Date"])
        db_max_date = df_hist["Date"].max()
        expected_symbols = set(df_hist["Symbol"].astype(str))

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        quote_failures = []
        nifty_response = fetch_retry(
            lambda: kite.quote(["NSE:NIFTY 50"]),
            "NIFTY quote",
            quote_failures,
        )
        nifty_data = nifty_response.get("NSE:NIFTY 50") if nifty_response else None
        if not nifty_data:
            die("Could not retrieve NIFTY 50 quote.")

        latest_trade_date = normalize_date(nifty_data.get("last_trade_time"))
        if not valid_date(latest_trade_date):
            die("NIFTY last_trade_time is missing or invalid.")
        if latest_trade_date > today:
            die("NIFTY reports a future trade date. Aborting.")
        if fetch_today and latest_trade_date < today:
            die(
                f"Today's completed market session is not confirmed. "
                f"Latest NIFTY trade: {latest_trade_date.date()}."
            )

        nifty_token = nifty_data.get("instrument_token")
        if not nifty_token:
            die("NIFTY instrument token is missing.")

        calendar_failures = []
        calendar_start = db_max_date - datetime.timedelta(days=CALENDAR_LOOKBACK_DAYS)
        calendar = fetch_retry(
            lambda: kite.historical_data(
                nifty_token,
                calendar_start.strftime("%Y-%m-%d"),
                latest_trade_date.strftime("%Y-%m-%d"),
                "day",
            ),
            "NIFTY calendar",
            calendar_failures,
        )
        if not calendar or calendar_failures:
            die("Could not retrieve a reliable NIFTY trading calendar.")

        market_dates = sorted({
            normalize_date(candle.get("date"))
            for candle in calendar
            if valid_date(normalize_date(candle.get("date")))
        })
        if not market_dates or market_dates[-1] != latest_trade_date:
            die("NIFTY history does not confirm the latest completed trading date.")

        required_dates = [
            date for date in market_dates
            if db_max_date < date <= latest_trade_date
        ]
        missing_dates = sorted(set(required_dates) - db_dates)
        print(
            f"Mode={mode}; database_last={db_max_date.date()}; "
            f"latest_market_date={latest_trade_date.date()}; "
            f"missing_dates={[date.date().isoformat() for date in missing_dates]}"
        )

        backfill_rows = []
        backfill_failures = []
        if missing_dates:
            start_missing = min(missing_dates)
            end_missing = max(missing_dates)
            for symbol in sorted(expected_symbols):
                token = get_token(kite, symbol, backfill_failures)
                if not token:
                    continue
                candles = fetch_retry(
                    lambda token=token: kite.historical_data(
                        token,
                        start_missing.strftime("%Y-%m-%d"),
                        end_missing.strftime("%Y-%m-%d"),
                        "day",
                    ),
                    f"backfill {symbol}",
                    backfill_failures,
                )
                temp = candles_to_df(candles, symbol)
                temp = temp[temp["Date"].isin(missing_dates)]
                temp = temp[temp.apply(is_valid_ohlcv, axis=1)]
                backfill_rows.extend(temp.to_dict("records"))
                returned_dates = set(temp["Date"])
                for date in set(missing_dates) - returned_dates:
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Date": date,
                        "Reason": "missing_backfill_candle",
                    })
                time.sleep(HISTORY_SLEEP_SECONDS)

        backfill_df = pd.DataFrame(backfill_rows, columns=REQUIRED_COLUMNS)
        expected_backfill = {
            (symbol, date)
            for symbol in expected_symbols
            for date in missing_dates
            if not (fetch_today and date == latest_trade_date)
        }
        actual_backfill = (
            set(zip(backfill_df["Symbol"], backfill_df["Date"]))
            if not backfill_df.empty else set()
        )
        missing_pairs = expected_backfill - actual_backfill
        if backfill_failures or missing_pairs:
            write_jsonl(
                AUDIT_DIR / f"backfill_failures_{now.strftime('%Y%m%d_%H%M%S')}.jsonl",
                backfill_failures + [
                    {"Symbol": symbol, "Date": date, "Reason": "required_pair_missing"}
                    for symbol, date in sorted(missing_pairs)
                ],
            )
            die(
                f"Backfill incomplete: failures={len(backfill_failures)}, "
                f"missing_pairs={len(missing_pairs)}. Database unchanged."
            )
        if not backfill_df.empty:
            validate_dataframe(backfill_df, "Backfill data")

        working = pd.concat([df_hist, backfill_df], ignore_index=True)
        working = (
            working.sort_values(["Symbol", "Date"])
            .drop_duplicates(["Symbol", "Date"], keep="last")
        )

        latest_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        if fetch_today:
            latest_rows = []
            latest_failures = []
            symbols = sorted(expected_symbols)
            for start in range(0, len(symbols), QUOTE_CHUNK_SIZE):
                chunk = symbols[start:start + QUOTE_CHUNK_SIZE]
                result = fetch_retry(
                    lambda chunk=chunk: kite.quote([f"NSE:{symbol}" for symbol in chunk]),
                    f"latest quotes {start // QUOTE_CHUNK_SIZE + 1}",
                    latest_failures,
                )
                if result:
                    for instrument_key, data in result.items():
                        symbol = instrument_key.replace("NSE:", "", 1)
                        if symbol not in expected_symbols:
                            continue
                        ohlc = data.get("ohlc") or {}
                        latest_rows.append({
                            "Date": latest_trade_date,
                            "Open": ohlc.get("open", 0),
                            "High": ohlc.get("high", 0),
                            "Low": ohlc.get("low", 0),
                            "Close": data.get("last_price", 0),
                            "Volume": data.get("volume", 0),
                            "Symbol": symbol,
                        })
                time.sleep(QUOTE_SLEEP_SECONDS)

            latest_df = pd.DataFrame(latest_rows, columns=REQUIRED_COLUMNS)
            latest_df = latest_df[latest_df.apply(is_valid_ohlcv, axis=1)]
            latest_df = latest_df.drop_duplicates(["Symbol", "Date"], keep="last")
            latest_symbols = set(latest_df["Symbol"])
            missing_latest = expected_symbols - latest_symbols
            critical = {
                symbol.strip()
                for symbol in os.environ.get(
                    "EOD_CRITICAL_SYMBOLS",
                    "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS",
                ).split(",")
                if symbol.strip()
            }
            if critical & missing_latest:
                die(f"Critical latest symbols missing: {sorted(critical & missing_latest)}")
            latest_limit = max(1, int(LATEST_MISSING_LIMIT * len(expected_symbols)))
            if latest_failures or len(missing_latest) > latest_limit:
                die(
                    f"Latest quote fetch incomplete: failures={len(latest_failures)}, "
                    f"missing={len(missing_latest)}, limit={latest_limit}."
                )
            validate_dataframe(latest_df, "Latest EOD data")
            working = working[working["Date"] != latest_trade_date]
            working = pd.concat([working, latest_df], ignore_index=True)

        validate_dataframe(working, "Combined database")
        working = (
            working.sort_values(["Symbol", "Date"])
            .drop_duplicates(["Symbol", "Date"], keep="last")
        )

        TMP_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        working.to_parquet(TMP_PARQUET, index=False)
        check_df = pd.read_parquet(TMP_PARQUET)
        validate_dataframe(check_df, "Read-back database")
        check_keys = (
            check_df[["Symbol", "Date"]]
            .sort_values(["Symbol", "Date"])
            .reset_index(drop=True)
        )
        working_keys = (
            working[["Symbol", "Date"]]
            .sort_values(["Symbol", "Date"])
            .reset_index(drop=True)
        )
        if not check_keys.equals(working_keys):
            die("Read-back Symbol/Date keys differ from working database.")

        if sha256_file(PARQUET_FILE) != source_hash:
            die("Source parquet changed before backup. Aborting.")

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / (
            f"nse_6yr_historical.backup."
            f"{now.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        backup_tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        shutil.copy2(PARQUET_FILE, backup_tmp)
        if sha256_file(backup_tmp) != source_hash:
            backup_tmp.unlink(missing_ok=True)
            die("Backup hash mismatch. Original database untouched.")
        os.replace(backup_tmp, backup_path)

        os.replace(TMP_PARQUET, PARQUET_FILE)
        print(
            f"✅ Saved successfully. mode={mode}; "
            f"recovered_dates={len(missing_dates)}; "
            f"today_fetched={fetch_today}"
        )

        exit_code = os.system("python analyze_6yr_data.py")
        if exit_code != 0:
            die(
                f"❌ analyze_6yr_data.py failed with exit code {exit_code}. "
                f"Database updated; backup preserved at {backup_path}."
            )

        write_json(
            AUDIT_DIR / f"eod_run_{now.strftime('%Y%m%d_%H%M%S')}.json",
            {
                "mode": mode,
                "database_before": str(db_max_date.date()),
                "latest_market_date": str(latest_trade_date.date()),
                "recovered_dates": [str(date.date()) for date in missing_dates],
                "backfill_rows": len(backfill_df),
                "today_fetched": fetch_today,
                "backup": str(backup_path),
            },
        )

    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()
