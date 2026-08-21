import os
import sys
import time
import pyotp
import json
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
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        
        try:
            # 1. Navigate to Kite Connect Login
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
            page.goto(login_url, wait_until="networkidle")
            
            # 2. Fill User ID and Password
            page.wait_for_selector("input[type='text']", timeout=10000)
            page.fill("input[type='text']", user_id)
            page.fill("input[type='password']", password)
            page.click("button[type='submit']")
            
            # 3. Wait for 2FA Screen and fill TOTP
            page.wait_for_selector("input[type='text']", timeout=10000)
            time.sleep(1) # Brief pause for UI render
            totp_token = pyotp.TOTP(totp_secret).now()
            
            # Playwright types the TOTP directly into the auto-focused 2FA input boxes
            page.keyboard.type(totp_token)
            
            # 4. Extract the Redirect URL Token
            request_token = None
            for _ in range(15):
                current_url = page.url
                if "request_token=" in current_url:
                    request_token = parse_qs(urlparse(current_url).query).get("request_token", [None])[0]
                    break
                time.sleep(1)
                
            browser.close()
            return request_token
            
        except Exception as e:
            print(f"❌ Browser Automation Error: {e}")
            browser.close()
            sys.exit(1)

# Execute Browser Login
req_token = get_request_token()

if not req_token:
    print("❌ Failed to extract Request Token from URL.")
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
        print(f"❌ Failed to upload secret to GitHub: {upload_res.text}")
        sys.exit(1)

except Exception as e:
    print("❌ Error communicating with GitHub API:", e)
    sys.exit(1)
