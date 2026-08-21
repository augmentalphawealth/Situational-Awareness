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

api_key = os.environ.get("KITE_API_KEY")
api_secret = os.environ.get("KITE_API_SECRET")
user_id = os.environ.get("KITE_USER_ID")
password = os.environ.get("KITE_PASSWORD")
totp_secret = os.environ.get("KITE_TOTP")
gh_pat = os.environ.get("GH_PAT")
repo = os.environ.get("GITHUB_REPOSITORY")

print("Initiating Daily Zerodha Token Rotation via Browser Automation...")

def get_request_token():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        
        try:
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
            page.goto(login_url, wait_until="domcontentloaded")
            
            # 1. Login Screen
            page.wait_for_selector("input[type='text']", timeout=15000)
            page.fill("input[type='text']", user_id)
            page.fill("input[type='password']", password)
            page.click("button[type='submit']")
            
            # 2. Two-Factor Authentication (2FA) Screen
            # Wait for the password field to disappear to confirm 2FA screen has loaded
            page.wait_for_selector("input[type='password']", state="hidden", timeout=15000)
            time.sleep(2)  # Critical pause for the 2FA DOM to render
            
            totp_token = pyotp.TOTP(totp_secret).now()
            
            # Fault-tolerant 2FA input targeting the morphed number field
            try:
                totp_input = page.locator("input[type='number'], input[label*='TOTP']").first
                totp_input.fill(totp_token)
            except Exception:
                # Fallback: If elements change again, rely on Kite's auto-focus and type directly
                page.keyboard.type(totp_token)
                
            # Click submit if Kite doesn't auto-submit the 6 digits
            try:
                submit_2fa = page.locator("button[type='submit']")
                if submit_2fa.is_visible(timeout=2000):
                    submit_2fa.click()
            except:
                pass
            
            # 3. Wait for redirect URL or handle the App Authorization screen
            request_token = None
            for _ in range(25):
                current_url = page.url
                if "request_token=" in current_url:
                    request_token = parse_qs(urlparse(current_url).query).get("request_token", [None])[0]
                    break
                
                # If Zerodha asks to authorize the API App, click the button
                try:
                    auth_btn = page.locator("button:has-text('Authorize'), button:has-text('I understand'), button:has-text('Accept')")
                    if auth_btn.count() > 0 and auth_btn.first.is_visible():
                        auth_btn.first.click()
                except:
                    pass
                    
                time.sleep(1)
            
            if not request_token:
                print(f"❌ Timeout reached. Final URL was: {page.url}")
                
            browser.close()
            return request_token
            
        except Exception as e:
            print(f"❌ Browser Automation Error: {e}")
            browser.close()
            sys.exit(1)

# Execute Browser Login
req_token = get_request_token()

if not req_token:
    print("❌ Failed to extract Request Token from URL. Check if Zerodha updated their login page.")
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
    if not gh_pat:
        print("❌ GH_PAT missing from environment variables.")
        sys.exit(1)
        
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
        print(f"❌ Failed to upload secret to GitHub: {upload_res.text}")
        sys.exit(1)

except Exception as e:
    print("❌ Error communicating with GitHub API:", e)
    sys.exit(1)
