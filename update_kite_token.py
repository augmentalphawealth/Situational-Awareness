import os
import sys
import time
import pyotp
import base64
import requests
import nacl.encoding
import nacl.public
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

# 1. Fail-Fast Environment Check
required_vars = {
    "KITE_API_KEY": os.environ.get("KITE_API_KEY"),
    "KITE_API_SECRET": os.environ.get("KITE_API_SECRET"),
    "KITE_USER_ID": os.environ.get("KITE_USER_ID"),
    "KITE_PASSWORD": os.environ.get("KITE_PASSWORD"),
    "KITE_TOTP": os.environ.get("KITE_TOTP"),
    "GH_PAT": os.environ.get("GH_PAT"),
    "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY")
}

missing = [name for name, val in required_vars.items() if not val]
if missing:
    print(f"❌ Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

api_key = required_vars["KITE_API_KEY"]
api_secret = required_vars["KITE_API_SECRET"]
user_id = required_vars["KITE_USER_ID"]
password = required_vars["KITE_PASSWORD"]
totp_secret = required_vars["KITE_TOTP"]
gh_pat = required_vars["GH_PAT"]
repo = required_vars["GITHUB_REPOSITORY"]

print("Initiating Daily Zerodha Token Rotation via Browser Automation...")

def token_from_url(url):
    """Safely extracts the request token from a URL string."""
    return parse_qs(urlparse(url).query).get("request_token", [None])[0]

def get_request_token():
    extracted_token = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()

        def intercept_request(request):
            if "request_token=" in request.url:
                token = token_from_url(request.url)
                if token and token not in extracted_token:
                    extracted_token.append(token)

        page.on("request", intercept_request)
        
        try:
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
            page.goto(login_url, wait_until="domcontentloaded")
            
            # Login Screen
            page.wait_for_selector("#userid", timeout=15000)
            page.fill("#userid", user_id)
            page.fill("#password", password)
            page.click("button[type='submit']")
            
            # 2FA Screen: Explicitly switch to TOTP mode
            time.sleep(3) 
            
            selectors_to_click = [
                "text=External Authenticator",
                "text=Use TOTP",
                "text=TOTP",
                "text=Problem with App Code"
            ]
            
            for selector in selectors_to_click:
                try:
                    locator = page.locator(selector).first
                    if locator.is_visible(timeout=2000):
                        locator.click()
                        time.sleep(1)
                        # If we clicked 'Problem with App Code', we must click 'External Authenticator' next
                        if "App Code" in selector:
                            ext_auth = page.locator("text=External Authenticator").first
                            if ext_auth.is_visible(timeout=2000):
                                ext_auth.click()
                                time.sleep(1)
                        break
                except Exception:
                    continue
                    
            # Find the exact TOTP input box
            totp_selectors = [
                "input[autocomplete='one-time-code']",
                "input[name*='totp' i]",
                "input[placeholder*='TOTP' i]",
                "input[type='number']",
                "input[type='text']"
            ]
            
            totp_input = None
            for selector in totp_selectors:
                try:
                    candidate = page.locator(selector).first
                    if candidate.is_visible(timeout=2000):
                        totp_input = candidate
                        break
                except Exception:
                    continue
                    
            if not totp_input:
                raise RuntimeError("Visible TOTP input box was not found on screen.")
                
            totp_token = pyotp.TOTP(totp_secret).now()
            totp_input.fill(totp_token)
            
            # Click submit if available, otherwise let Zerodha auto-submit
            try:
                submit_2fa = page.locator("button[type='submit']")
                if submit_2fa.is_visible(timeout=2000):
                    submit_2fa.click()
            except Exception:
                pass
            
            # Wait for the network listener to catch the token
            for _ in range(25):
                if extracted_token:
                    break
                try:
                    # Clear final app authorization screen if prompted
                    auth_btn = page.locator("button:has-text('Authorize'), button:has-text('I understand'), button:has-text('Accept')")
                    if auth_btn.count() > 0 and auth_btn.first.is_visible():
                        auth_btn.first.click()
                except Exception:
                    pass
                time.sleep(1)
            
            if not extracted_token:
                current_token = token_from_url(page.url)
                if current_token:
                    extracted_token.append(current_token)
                else:
                    print("❌ Timeout reached before request token was captured. (URL redacted for security)")
                    
        except Exception as e:
            # Crash Rescue: If localhost connection refused, the URL still contains the token
            current_token = token_from_url(page.url)
            if current_token and current_token not in extracted_token:
                extracted_token.append(current_token)
                
            if extracted_token:
                print("✅ Redirect navigation crashed (expected on GitHub Actions), but request token was successfully captured.")
            else:
                print(f"❌ Browser automation error: {type(e).__name__}: {e}")
                
        finally:
            # Close safely to avoid ghost processes
            browser.close()
            
    return extracted_token[0] if extracted_token else None

# Execute Browser Login
req_token = get_request_token()

if not req_token:
    print("❌ Failed to extract Request Token. Automation aborted.")
    sys.exit(1)

# --- Exchange Request Token for Access Token ---
print("✅ Extracted Request Token. Generating Access Token...")
try:
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(req_token, api_secret=api_secret)
    new_access_token = data["access_token"]
    print("✅ Successfully generated fresh Zerodha Access Token.")
except Exception as e:
    print(f"❌ Error exchanging token with Zerodha: {e}")
    sys.exit(1)

# --- GitHub Secret Injection Phase (PyNaCl Encryption) ---
print("Injecting Token into GitHub Secrets...")
try:
    headers = {
        "Authorization": f"Bearer {gh_pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    pub_key_res = requests.get(f"https://api.github.com/repos/{repo}/actions/secrets/public-key", headers=headers)
    if pub_key_res.status_code != 200:
        print(f"❌ Failed to fetch GitHub public key: {pub_key_res.text}")
        sys.exit(1)
    
    pub_key_data = pub_key_res.json()
    key_id = pub_key_data['key_id']
    public_key = pub_key_data['key']

    public_key_bytes = nacl.public.PublicKey(public_key.encode("utf-8"), nacl.encoding.Base64Encoder())
    sealed_box = nacl.public.SealedBox(public_key_bytes)
    encrypted_token_bytes = sealed_box.encrypt(new_access_token.encode("utf-8"))
    encrypted_token = base64.b64encode(encrypted_token_bytes).decode("utf-8")

    payload = {"encrypted_value": encrypted_token, "key_id": key_id}
    upload_res = requests.put(f"https://api.github.com/repos/{repo}/actions/secrets/KITE_ACCESS_TOKEN", headers=headers, json=payload)
    
    if upload_res.status_code in [201, 204]:
        print("✅ KITE_ACCESS_TOKEN secret successfully updated in GitHub.")
    else:
        print(f"❌ Failed to upload secret to GitHub: {upload_res.status_code} - {upload_res.text}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error communicating with GitHub API: {e}")
    sys.exit(1)
