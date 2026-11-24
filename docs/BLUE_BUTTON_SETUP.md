# CMS Blue Button 2.0 setup (synthetic Medicare claims)

Blue Button 2.0 is the CMS API that lets Medicare enrollees share their Part A, B and D claims with apps they
trust. Its **sandbox** is open to everyone and contains only **synthetic** enrollees, so BenefitIQ can show
claims history without any real patient data. This step is optional - BenefitIQ works without it.

## 1. Create a sandbox app (5 minutes)
1. Open https://sandbox.bluebutton.cms.gov and create a free developer account (verify your email).
2. On the dashboard click **Add an Application**:
   - Name: `BenefitIQ (portfolio)`
   - Client type: **Confidential** (the only option) - Grant type: **Authorization code**
   - Callback / Redirect URI: `http://localhost:8765/callback`
   - Data access: choose the longest access duration offered (e.g. 13 months) so you get **refresh tokens**
3. Save. Copy the **Client ID** and **Client Secret** (later: dashboard -> View/Edit App).

## 2. Add the credentials to `.env`
```
BLUEBUTTON_CLIENT_ID=...
BLUEBUTTON_CLIENT_SECRET=...
BLUEBUTTON_REDIRECT_URI=http://localhost:8765/callback
```
The redirect URI must match the app exactly (including the trailing path).

## 3. Connect a BenefitIQ member to a synthetic Medicare enrollee
```
python scripts/connect_bluebutton.py --member M1001
```
1. Open the printed URL and log in with a **synthetic** sandbox user, e.g. `BBUser00000` / `PW00000!`
   (other users follow the same pattern; CMS publishes a list of synthetic users by claim count).
2. Approve access. The browser then goes to `http://localhost:8765/callback?...` - the page may not load.
3. Copy the **whole address** from the browser bar and paste it into the terminal.

Check or remove a connection:
```
python scripts/connect_bluebutton.py --member M1001 --status
python scripts/connect_bluebutton.py --member M1001 --disconnect
```
Tokens are saved in `data/bluebutton/M1001.json` (git-ignored, owner-only permissions) and are never logged.

## Security notes
- OAuth 2.0 authorization code with **PKCE (S256)** and a 256-bit `state`, as Blue Button requires.
- The app only ever sees synthetic enrollees in the sandbox. Production access requires a CMS review.
- BenefitIQ passes only minimum-necessary claim facts (dates, service codes and descriptions) to the agents.
- Blue Button v1/v2 are scheduled for deprecation in 2027 (v3 is live); the version is a setting
  (`BLUEBUTTON_API_VERSION`).
