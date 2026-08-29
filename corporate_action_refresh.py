import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from kite_helper import fetch_with_backoff, get_kite_client

IST = ZoneInfo("Asia/Kolkata")

PARQUET_FILE = Path("nse_6yr_historical.parquet")
TMP_PARQUET = Path("nse_6yr_historical.corporate_action.tmp.parquet")
LOCK_FILE = Path("corporate_action_refresh.lock")
BACKUP_DIR = Path("corporate_action_backups")
AUDIT_DIR = Path("corporate_action_audits")

REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
VERIFY_LOOKBACK_TRADING_DAYS = int(
    os.environ.get("CORPORATE_ACTION_VERIFY_DAYS", "5")
)
DIFFERENCE_THRESHOLD_PCT = float(
    os.environ.get("CORPORATE_ACTION_THRESHOLD_PCT", "2.0")
)
MAX_SYMBOL_REFRESHES = int(
    os.environ.get("CORPORATE_ACTION_MAX_SYMBOL_REFRESHES", "50")
)
BACKUP_RETENTION = int(os.environ.get("CORPORATE_ACTION_BACKUP_RETENTION", "14"))
MAX_KITE_HISTORY_DAYS = 1800


def die(message, code=1):
    print(message, flush=True)
    raise SystemExit(code)


def normalize_date(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def clean_symbol(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


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
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
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
            die("Existing corporate-action lock has no PID; inspect it manually.")
        try:
            pid = int(pid_text)
        except ValueError:
            die("Existing corporate-action lock has an invalid PID; inspect it manually.")
        if host and host != socket.gethostname():
            die(f"Lock belongs to another host ({host}); refusing automatic removal.")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            print(f"Removed stale lock for dead PID {pid}.", flush=True)
        except PermissionError:
            die(f"Cannot verify lock PID {pid}; aborting.")
        except OSError as exc:
            die(f"Could not verify lock PID {pid}: {exc}")
        else:
            die(f"Another corporate-action refresh is active: PID {pid}.")

    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                f"pid={os.getpid()}\n"
                f"host={socket.gethostname()}\n"
                f"time={datetime.datetime.now(IST).isoformat()}\n"
            )
    except FileExistsError:
        die("Another corporate-action refresh acquired the lock. Aborting.")


def release_lock(path):
    try:
        if not path.exists():
            return
        info = read_lock(path)
        if (
            info.get("pid") == str(os.getpid())
            and info.get("host") == socket.gethostname()
        ):
            path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Warning: lock cleanup failed: {exc}", flush=True)


def call_kite(fetch_func, label, failures):
    result = fetch_with_backoff(fetch_func)
    if result is None:
        failures.append({"label": label, "reason": "exhausted_retries"})
    return result


def candles_to_df(candles, symbol):
    if not candles:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(candles).rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    if set(REQUIRED_COLUMNS[1:]) - set(df.columns):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df["Symbol"] = clean_symbol(symbol)
    df["Date"] = df["Date"].apply(normalize_date)
    return df[REQUIRED_COLUMNS].copy()


def valid_ohlcv(df):
    numeric = df[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        numeric["Open"].gt(0)
        & numeric["High"].gt(0)
        & numeric["Low"].gt(0)
        & numeric["Close"].gt(0)
        & numeric["Volume"].ge(0)
        & numeric["High"].ge(numeric[["Open", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "Close"]].min(axis=1))
        & numeric["High"].ge(numeric["Low"])
    )
    return valid.fillna(False)


def validate_database(df, label):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        die(f"{label} is missing columns: {sorted(missing)}")
    if df["Date"].isna().any():
        die(f"{label} contains invalid dates.")
    if (df["Symbol"] == "").any():
        die(f"{label} contains blank symbols.")
    if df.duplicated(["Symbol", "Date"]).any():
        die(f"{label} contains duplicate Symbol/Date keys.")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def rotate_backups():
    backups = sorted(
        BACKUP_DIR.glob("nse_6yr_historical.before_corporate_action.*.parquet"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old_backup in backups[BACKUP_RETENTION:]:
        try:
            old_backup.unlink()
        except OSError as exc:
            print(f"Warning: could not delete old backup {old_backup}: {exc}", flush=True)


def latest_verify_dates(df):
    dates = sorted(df["Date"].dropna().unique())
    if not dates:
        die("Database has no usable dates.")
    return [pd.Timestamp(date).normalize() for date in dates[-VERIFY_LOOKBACK_TRADING_DAYS:]]


def get_current_token(kite, symbol, failures):
    quote = call_kite(
        lambda: kite.quote([f"NSE:{symbol}"]),
        f"instrument token {symbol}",
        failures,
    )
    data = quote.get(f"NSE:{symbol}") if quote else None
    return data.get("instrument_token") if data else None


def fetch_history_window(kite, token, symbol, start_date, end_date, failures):
    candles = call_kite(
        lambda: kite.historical_data(
            token,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            "day",
        ),
        f"historical data {symbol} {start_date.date()} to {end_date.date()}",
        failures,
    )
    df = candles_to_df(candles, symbol)
    if df.empty:
        return df
    df = df[df["Date"].notna()].copy()
    return df[valid_ohlcv(df)].copy()


def fetch_full_history_in_chunks(kite, token, symbol, start_date, end_date, failures):
    """Fetch a complete symbol history without exceeding Kite's 2,000-day limit."""
    chunks = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + datetime.timedelta(days=MAX_KITE_HISTORY_DAYS - 1),
            end_date,
        )
        chunk = fetch_history_window(
            kite,
            token,
            symbol,
            chunk_start,
            chunk_end,
            failures,
        )
        if chunk.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        chunks.append(chunk)
        chunk_start = chunk_end + datetime.timedelta(days=1)

    full_history = pd.concat(chunks, ignore_index=True)
    full_history = (
        full_history.sort_values(["Symbol", "Date"])
        .drop_duplicates(["Symbol", "Date"], keep="last")
        .reset_index(drop=True)
    )
    return full_history


def compare_same_dates(stored, fresh, verify_dates):
    stored_recent = stored[stored["Date"].isin(verify_dates)][["Symbol", "Date", "Close"]].copy()
    fresh_recent = fresh[fresh["Date"].isin(verify_dates)][["Symbol", "Date", "Close"]].copy()
    if stored_recent.empty or fresh_recent.empty:
        return pd.DataFrame()

    stored_recent = stored_recent.rename(columns={"Close": "Stored_Close"})
    fresh_recent = fresh_recent.rename(columns={"Close": "Fresh_Close"})
    comparison = stored_recent.merge(
        fresh_recent,
        on=["Symbol", "Date"],
        how="inner",
    )
    comparison["Stored_Close"] = pd.to_numeric(comparison["Stored_Close"], errors="coerce")
    comparison["Fresh_Close"] = pd.to_numeric(comparison["Fresh_Close"], errors="coerce")
    comparison = comparison[
        comparison["Stored_Close"].gt(0) & comparison["Fresh_Close"].gt(0)
    ].copy()
    comparison["Difference_Pct"] = (
        (comparison["Fresh_Close"] - comparison["Stored_Close"])
        .abs()
        .div(comparison["Stored_Close"])
        .mul(100)
    )
    return comparison[comparison["Difference_Pct"] >= DIFFERENCE_THRESHOLD_PCT].copy()


def main():
    if VERIFY_LOOKBACK_TRADING_DAYS < 1:
        die("CORPORATE_ACTION_VERIFY_DAYS must be at least 1.")
    if DIFFERENCE_THRESHOLD_PCT <= 0:
        die("CORPORATE_ACTION_THRESHOLD_PCT must be greater than zero.")
    if MAX_SYMBOL_REFRESHES < 1:
        die("CORPORATE_ACTION_MAX_SYMBOL_REFRESHES must be at least 1.")

    now = datetime.datetime.now(IST)
    acquire_lock(LOCK_FILE)
    try:
        if TMP_PARQUET.exists():
            die(f"Temporary file exists: {TMP_PARQUET}. Inspect it first.")
        if not PARQUET_FILE.exists():
            die(f"Master database not found: {PARQUET_FILE}")

        source_hash = sha256_file(PARQUET_FILE)
        if source_hash is None:
            die("Could not hash master parquet.")

        stored = pd.read_parquet(PARQUET_FILE)
        if sha256_file(PARQUET_FILE) != source_hash:
            die("Master parquet changed while being loaded.")

        stored["Symbol"] = stored["Symbol"].map(clean_symbol)
        stored["Date"] = stored["Date"].apply(normalize_date)
        validate_database(stored, "Master database")

        verify_dates = latest_verify_dates(stored)
        full_start_date = stored["Date"].min()
        full_end_date = stored["Date"].max()
        symbols = sorted(stored["Symbol"].unique())

        print(
            "Corporate-action scan: "
            f"symbols={len(symbols)}; "
            f"verify_dates={[date.date().isoformat() for date in verify_dates]}; "
            f"threshold={DIFFERENCE_THRESHOLD_PCT:.2f}%.",
            flush=True,
        )

        kite = get_kite_client()
        scan_failures = []
        candidates = []
        candidate_details = []

        for symbol in symbols:
            token = get_current_token(kite, symbol, scan_failures)
            if not token:
                continue

            stored_symbol = stored[stored["Symbol"] == symbol].copy()
            recent_fresh = fetch_history_window(
                kite,
                token,
                symbol,
                min(verify_dates),
                max(verify_dates),
                scan_failures,
            )
            differences = compare_same_dates(stored_symbol, recent_fresh, verify_dates)
            if differences.empty:
                continue

            candidates.append(symbol)
            candidate_details.extend(differences.to_dict("records"))
            print(
                f"Candidate {symbol}: maximum same-date close difference "
                f"{differences['Difference_Pct'].max():.2f}%.",
                flush=True,
            )
            if len(set(candidates)) > MAX_SYMBOL_REFRESHES:
                die(
                    f"Detected more than {MAX_SYMBOL_REFRESHES} candidates. "
                    "Refusing a broad automatic rewrite; inspect the audit output."
                )

        audit_stamp = now.strftime("%Y%m%d_%H%M%S")
        candidate_df = pd.DataFrame(candidate_details)
        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(["Symbol", "Date"])
        write_csv(
            AUDIT_DIR / f"corporate_action_candidates_{audit_stamp}.csv",
            candidate_df,
        )
        write_json(
            AUDIT_DIR / f"corporate_action_scan_{audit_stamp}.json",
            {
                "run_time_ist": now.isoformat(),
                "verify_dates": [date.date().isoformat() for date in verify_dates],
                "threshold_pct": DIFFERENCE_THRESHOLD_PCT,
                "symbols_scanned": len(symbols),
                "candidate_symbols": sorted(set(candidates)),
                "candidate_count": len(set(candidates)),
                "scan_failure_count": len(scan_failures),
                "scan_failures": scan_failures,
            },
        )

        if not candidates:
            print(
                "No same-date close differences at or above the threshold. "
                "Database unchanged.",
                flush=True,
            )
            return

        refresh_failures = []
        refreshed_histories = {}
        for symbol in sorted(set(candidates)):
            token = get_current_token(kite, symbol, refresh_failures)
            if not token:
                refresh_failures.append({"Symbol": symbol, "reason": "refresh_token_unavailable"})
                continue

            history = fetch_full_history_in_chunks(
                kite,
                token,
                symbol,
                full_start_date,
                full_end_date,
                refresh_failures,
            )
            if history.empty:
                refresh_failures.append({"Symbol": symbol, "reason": "empty_full_history"})
                continue
            if history.duplicated(["Symbol", "Date"]).any():
                refresh_failures.append({"Symbol": symbol, "reason": "duplicate_full_history"})
                continue
            refreshed_histories[symbol] = history

        if refresh_failures:
            write_json(
                AUDIT_DIR / f"corporate_action_refresh_failures_{audit_stamp}.json",
                {"run_time_ist": now.isoformat(), "failures": refresh_failures},
            )

        candidate_symbols = set(candidates)
        if candidate_symbols - set(refreshed_histories):
            die(
                "At least one detected candidate could not be fully refreshed. "
                "Database unchanged; inspect corporate_action_audits/."
            )

        replacement_symbols = set(refreshed_histories)
        unchanged = stored[~stored["Symbol"].isin(replacement_symbols)].copy()
        replacement = pd.concat(list(refreshed_histories.values()), ignore_index=True)
        combined = pd.concat([unchanged, replacement], ignore_index=True)
        combined["Symbol"] = combined["Symbol"].map(clean_symbol)
        combined["Date"] = combined["Date"].apply(normalize_date)
        combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        validate_database(combined, "Corporate-action refreshed database")

        combined.to_parquet(TMP_PARQUET, index=False)
        check = pd.read_parquet(TMP_PARQUET)
        check["Symbol"] = check["Symbol"].map(clean_symbol)
        check["Date"] = check["Date"].apply(normalize_date)
        validate_database(check, "Temporary corporate-action parquet")

        if sha256_file(PARQUET_FILE) != source_hash:
            die("Master parquet changed before replacement. Aborting.")

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / (
            f"nse_6yr_historical.before_corporate_action.{audit_stamp}.parquet"
        )
        backup_temp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        shutil.copy2(PARQUET_FILE, backup_temp)
        if sha256_file(backup_temp) != source_hash:
            backup_temp.unlink(missing_ok=True)
            die("Backup hash mismatch. Master parquet remains untouched.")
        os.replace(backup_temp, backup_path)
        os.replace(TMP_PARQUET, PARQUET_FILE)
        rotate_backups()

        result = subprocess.run([sys.executable, "analyze_6yr_data.py"], check=False)
        if result.returncode != 0:
            die(
                f"analyze_6yr_data.py failed with exit code {result.returncode}. "
                f"Master parquet was refreshed; backup is {backup_path}."
            )

        write_json(
            AUDIT_DIR / f"corporate_action_refresh_{audit_stamp}.json",
            {
                "run_time_ist": now.isoformat(),
                "threshold_pct": DIFFERENCE_THRESHOLD_PCT,
                "verify_dates": [date.date().isoformat() for date in verify_dates],
                "refreshed_symbols": sorted(replacement_symbols),
                "refreshed_symbol_count": len(replacement_symbols),
                "backup": str(backup_path),
                "scan_failure_count": len(scan_failures),
                "refresh_failure_count": len(refresh_failures),
                "max_kite_history_days": MAX_KITE_HISTORY_DAYS,
            },
        )
        print(
            f"Corporate-action refresh completed for {len(replacement_symbols)} symbol(s).",
            flush=True,
        )
    finally:
        TMP_PARQUET.unlink(missing_ok=True)
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()
