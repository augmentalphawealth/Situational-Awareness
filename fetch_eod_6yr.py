import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from kiteconnect import KiteConnect


IST = ZoneInfo("Asia/Kolkata")
EOD_TIME = datetime.time(17, 0)
QUOTE_CHUNK_SIZE = 200
MAX_RETRIES = 5
QUOTE_SLEEP_SECONDS = 1.1
HISTORY_SLEEP_SECONDS = 1.5
CALENDAR_LOOKBACK_DAYS = 15
BACKUP_RETENTION = 14

PARQUET_FILE = Path("nse_6yr_historical.parquet")
TMP_PARQUET = Path("nse_6yr_historical.tmp.parquet")
LOCK_FILE = Path("eod_sync.lock")
BACKUP_DIR = Path("eod_backups")
AUDIT_DIR = Path("eod_audits")

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


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


def read_lock(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    values[key] = value
    except OSError as exc:
        die(f"Cannot read lock file: {exc}")
    return values


def acquire_lock(path):
    if path.exists():
        info = read_lock(path)
        pid_text = info.get("pid")
        host = info.get("host")
        if not pid_text:
            die("Lock file has no valid PID. Inspect it manually.")
        try:
            pid = int(pid_text)
        except ValueError:
            die("Lock file contains an invalid PID. Inspect it manually.")
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
        info = read_lock(path)
        if (
            info.get("pid") == str(os.getpid())
            and info.get("host") == socket.gethostname()
        ):
            path.unlink(missing_ok=True)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"Lock cleanup failed: {exc}")


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


def is_valid_new_ohlcv(row):
    try:
        return (
            float(row["Open"]) > 0
            and float(row["High"]) > 0
            and float(row["Low"]) > 0
            and float(row["Close"]) > 0
            and float(row["Volume"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_existing_structure(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"Existing database missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die("Existing database contains invalid dates.")
    if df.duplicated(["Symbol", "Date"]).any():
        die("Existing database contains duplicate Symbol/Date keys.")


def validate_new_data(df, name):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"{name} missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die(f"{name} contains invalid dates.")
    if df.duplicated(["Symbol", "Date"]).any():
        die(f"{name} contains duplicate Symbol/Date keys.")
    valid = df.apply(is_valid_new_ohlcv, axis=1)
    if not valid.all():
        die(f"{name} contains invalid newly fetched OHLCV rows.")


def candles_to_df(candles, symbol):
    if not candles:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.DataFrame(candles).rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    history_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if set(history_columns) - set(df.columns):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df["Date"] = df["Date"].apply(normalize_date)
    df["Symbol"] = symbol
    return (
        df[REQUIRED_COLUMNS]
        .sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
    )


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


def rotate_backups():
    backups = sorted(
        BACKUP_DIR.glob("nse_6yr_historical.backup.*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in backups[BACKUP_RETENTION:]:
        try:
            path.unlink()
        except OSError as exc:
            print(f"Warning: could not delete old backup {path}: {exc}")


def run_analyzer():
    return subprocess.run(
        [sys.executable, "analyze_6yr_data.py"],
        check=False,
    ).returncode


def main():
    now = datetime.datetime.now(IST)
    today = normalize_date(now)
    is_weekday = now.weekday() < 5
    fetch_today = is_weekday and now.time() >= EOD_TIME
    mode = "eod" if fetch_today else "recovery_only"

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

        validate_existing_structure(df_hist)
        df_hist["Date"] = df_hist["Date"].apply(normalize_date)
        db_dates = set(df_hist["Date"])
        db_max_date = df_hist["Date"].max()
        expected_symbols = set(df_hist["Symbol"].astype(str))

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        nifty_failures = []
        nifty_result = fetch_retry(
            lambda: kite.quote(["NSE:NIFTY 50"]),
            "NIFTY quote",
            nifty_failures,
        )
        nifty_data = nifty_result.get("NSE:NIFTY 50") if nifty_result else None
        if not nifty_data:
            die("Could not retrieve NIFTY 50 quote.")

        last_trade_date = normalize_date(nifty_data.get("last_trade_time"))
        nifty_token = nifty_data.get("instrument_token")
        if not nifty_token:
            die("NIFTY instrument token is missing.")

        # If last_trade_time is absent on a weekend/holiday, historical NIFTY data
        # supplies the last completed trading date. This is also a safety fallback.
        calendar_failures = []
        calendar_start = db_max_date - datetime.timedelta(days=CALENDAR_LOOKBACK_DAYS)
        calendar = fetch_retry(
            lambda: kite.historical_data(
                nifty_token,
                calendar_start.strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
                "day",
            ),
            "NIFTY trading calendar",
            calendar_failures,
        )
        if not calendar or calendar_failures:
            die("Could not retrieve a reliable NIFTY trading calendar.")

        market_dates = sorted({
            normalize_date(candle.get("date"))
            for candle in calendar
            if valid_date(normalize_date(candle.get("date")))
        })
        if not market_dates:
            die("NIFTY historical calendar returned no valid dates.")

        latest_calendar_date = market_dates[-1]
        if valid_date(last_trade_date) and last_trade_date > latest_calendar_date:
            latest_market_date = last_trade_date
        else:
            latest_market_date = latest_calendar_date

        if fetch_today and latest_market_date < today:
            die(
                f"Today's EOD session is not confirmed. "
                f"Latest completed market date: {latest_market_date.date()}."
            )

        required_dates = [
            date for date in market_dates
            if db_max_date < date <= latest_market_date
        ]
        missing_dates = sorted(set(required_dates) - db_dates)
        print(
            f"Mode={mode}; database_last={db_max_date.date()}; "
            f"latest_market_date={latest_market_date.date()}; "
            f"missing_dates={[date.date().isoformat() for date in missing_dates]}"
        )
        # NEW: Add backfill progress estimation
        print(f"Backfill needed: {len(expected_symbols)} symbols × {len(missing_dates)} dates = {len(expected_symbols) * len(missing_dates)} API calls")
        print(f"Estimated time: {len(expected_symbols) * len(missing_dates) * (HISTORY_SLEEP_SECONDS + 2) / 60:.1f} minutes")

        backfill_rows = []
        backfill_failures = []
        if missing_dates:
            start_missing = min(missing_dates)
            end_missing = max(missing_dates)
            for symbol in sorted(expected_symbols):
                token_failures = []
                token_response = fetch_retry(
                    lambda symbol=symbol: kite.quote([f"NSE:{symbol}"]),
                    f"token {symbol}",
                    token_failures,
                )
                token_data = token_response.get(f"NSE:{symbol}") if token_response else None
                token = token_data.get("instrument_token") if token_data else None
                if token_failures or not token:
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Reason": "instrument_token_failure",
                        "Failures": token_failures,
                    })
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
                if temp.empty:
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Reason": "empty_backfill_response",
                    })
                    continue
                temp = temp[temp.apply(is_valid_new_ohlcv, axis=1)]
                backfill_rows.extend(temp.to_dict("records"))
                returned_dates = set(temp["Date"])
                for date in sorted(set(missing_dates) - returned_dates):
                    backfill_failures.append({
                        "Symbol": symbol,
                        "Date": date,
                        "Reason": "missing_backfill_candle",
                    })
                time.sleep(HISTORY_SLEEP_SECONDS)

        backfill_df = pd.DataFrame(backfill_rows, columns=REQUIRED_COLUMNS)
        actual_pairs = (
            set(zip(backfill_df["Symbol"], backfill_df["Date"]))
            if not backfill_df.empty else set()
        )
        required_pairs = {
            (symbol, date)
            for symbol in expected_symbols
            for date in missing_dates
            if not (fetch_today and date == latest_market_date)
        }
        missing_pairs = required_pairs - actual_pairs
        
        # NEW: Log failures but continue instead of aborting
        if backfill_failures or missing_pairs:
            records = backfill_failures + [
                {"Symbol": symbol, "Date": date, "Reason": "required_pair_missing"}
                for symbol, date in sorted(missing_pairs)
            ]
            write_jsonl(
                AUDIT_DIR / f"backfill_failures_{now.strftime('%Y%m%d_%H%M%S')}.jsonl",
                records,
            )
            # NEW: Print warning but continue with available data
            print(f"⚠️ WARNING: Backfill had {len(backfill_failures)} failures and {len(missing_pairs)} missing pairs.")
            print(f"⚠️ Failing symbols logged to audit file. Continuing with available data...")
            # DO NOT abort - proceed with update

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
                            "Date": latest_market_date,
                            "Open": ohlc.get("open", 0),
                            "High": ohlc.get("high", 0),
                            "Low": ohlc.get("low", 0),
                            "Close": data.get("last_price", 0),
                            "Volume": data.get("volume", 0),
                            "Symbol": symbol,
                        })
                time.sleep(QUOTE_SLEEP_SECONDS)

            latest_df = pd.DataFrame(latest_rows, columns=REQUIRED_COLUMNS)
            latest_df = latest_df[latest_df.apply(is_valid_new_ohlcv, axis=1)]
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
            if critical & missing_latest or latest_failures or missing_latest:
                die(
                    f"Latest EOD fetch incomplete: failures={len(latest_failures)}, "
                    f"missing_symbols={len(missing_latest)}. Database unchanged."
                )
            validate_new_data(latest_df, "Latest EOD data")
            working = working[working["Date"] != latest_market_date]
            working = pd.concat([working, latest_df], ignore_index=True)

        if working.duplicated(["Symbol", "Date"]).any():
            die("Combined database contains duplicate Symbol/Date keys.")
        if working["Date"].isna().any():
            die("Combined database contains invalid dates.")

        working.to_parquet(TMP_PARQUET, index=False)
        check_df = pd.read_parquet(TMP_PARQUET)
        if check_df.empty or check_df["Date"].isna().any():
            die("Temporary database read-back validation failed.")
        if check_df.duplicated(["Symbol", "Date"]).any():
            die("Temporary database contains duplicate keys.")

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
            f"Saved successfully. mode={mode}; "
            f"recovered_dates={len(missing_dates)}; "
            f"today_fetched={fetch_today}"
        )

        result = subprocess.run(
            [sys.executable, "analyze_6yr_data.py"],
            check=False,
        )
        if result.returncode != 0:
            die(
                f"analyze_6yr_data.py failed with exit code {result.returncode}. "
                f"Database updated; backup preserved at {backup_path}."
            )

        write_json(
            AUDIT_DIR / f"eod_run_{now.strftime('%Y%m%d_%H%M%S')}.json",
            {
                "mode": mode,
                "database_before": str(db_max_date.date()),
                "latest_market_date": str(latest_market_date.date()),
                "recovered_dates": [str(date.date()) for date in missing_dates],
                "backfill_rows": len(backfill_df),
                "today_fetched": fetch_today,
                "backup": str(backup_path),
                "backfill_failures": len(backfill_failures),
                "missing_pairs": len(missing_pairs),
            },
        )

    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()