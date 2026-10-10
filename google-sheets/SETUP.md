# Connect the daily scan to Google Sheets

1. Open the destination Google Sheet and select **Extensions > Apps Script**.
2. Replace the editor contents with [`Code.gs`](Code.gs).
3. Generate a private token in the Codespaces terminal:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

4. Paste that value over `REPLACE_WITH_A_LONG_RANDOM_TOKEN` in `Code.gs`.
5. Select the `setup` function, click **Run**, and approve Google's prompts.
6. Select **Deploy > New deployment > Web app**. Set **Execute as** to **Me**,
   set access to **Anyone**, deploy, and copy the `/exec` web-app URL. The token
   is still required for every update.
7. In the GitHub repository, open **Settings > Secrets and variables > Actions**.
   Add these two repository secrets:

   - `GOOGLE_SHEETS_WEBHOOK_URL`: the `/exec` URL from step 6
   - `GOOGLE_SHEETS_TOKEN`: the exact token generated in step 3

8. Open **Actions > Daily DECA scan > Run workflow** once. After it completes,
   the Google Sheet will contain a `Shortlist` tab. The workflow then runs on
   weekdays at 3:45 p.m. New York time.

Keep the token private. To revoke access, delete the Apps Script deployment or
change the token in both Apps Script and the GitHub repository secret.
