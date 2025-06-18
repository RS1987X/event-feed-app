## 🗂 Data Management with DVC + Google Drive

This project uses [DVC (Data Version Control)](https://dvc.org/) to version-control and sync the `data/` folder via Google Drive.

### Setup (on a new machine)

```bash
git clone https://github.com/RS1987X/event-feed-app.git
cd event-feed-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "dvc[gdrive]"

Create a file called .dvc/config.local in the root directory (this is Git-ignored by default):

['remote "gdrive_remote"']
    gdrive_client_id = your-client-id.apps.googleusercontent.com
    gdrive_client_secret = your-client-secret


You can transfer this file manually from another machine (e.g., via USB) — do not commit it.

Then pull the data:
dvc pull

To track and push new or updated data:
dvc add data/
git commit -am "Add/update data"
git push
dvc push

Secrets like your OAuth client ID/secret are stored only in .dvc/config.local, which is automatically Git-ignored.

.dvc/config contains the remote URL and is safe to commit.

GitHub will block pushes if secrets are accidentally committed.

DVC uses .dvc/cache/ to store data versions — no need to manage this manually.