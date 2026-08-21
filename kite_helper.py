import os
import sys
import time
from kiteconnect import KiteConnect

_last_request_time = 0.0
MIN_INTERVAL = 0.35  # Guarantees < 3 requests per second globally

def pace():
    """Global token-bucket rate limiter."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_time = time.monotonic()

def get_kite_client():
    """Initializes Kite client using the Playwright-rotated access token."""
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    
    if not api_key or not access_token:
        print("❌ KITE_API_KEY or KITE_ACCESS_TOKEN missing from environment.")
        sys.exit(1)
        
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # Validate token is alive
        pace()
        kite.profile()
        return kite
    except Exception as e:
        print(f"❌ Kite Authentication Failed (Token may be expired): {type(e).__name__} - {e}")
        sys.exit(1)

def fetch_with_backoff(fetch_func, *args, **kwargs):
    """Executes a Kite API call with exponential backoff for rate limits."""
    for attempt in range(1, 6):
        pace()
        try:
            return fetch_func(*args, **kwargs)
        except Exception as exc:
            err_msg = str(exc)
            print(f"⚠️ API Error ({type(exc).__name__}): {err_msg}. Attempt {attempt}/5")
            if "429" in err_msg or "403" in err_msg or "Rate" in err_msg:
                delay = min(60, 2 ** attempt)
                time.sleep(delay)
            else:
                time.sleep(1)
    print("❌ All retry attempts failed for this chunk.")
    return None
