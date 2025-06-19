## 🗂 Data Management with DVC + Google Drive

This project uses [DVC (Data Version Control)](https://dvc.org/) to version-control and sync the `data/` folder via Google Drive.

### Setup (on a new machine)

```bash
#Clone and enter
git clone https://github.com/RS1987X/event-feed-app.git
cd event-feed-app
#Create & activate virtualenv
python3 -m venv venv
source venv/bin/activate
#install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

>Create a file called .dvc/config.local in the root directory (this is Git-ignored by default):

```bash
dvc remote modify --local gdrive_remote gdrive_client_id     <YOUR_CLIENT_ID>
dvc remote modify --local gdrive_remote gdrive_client_secret <YOUR_CLIENT_SECRET>
```

>Use the actual client id and client secret from the credentials file. This inserts the following lines in config.local (its created if it does not already exist)

['remote "gdrive_remote"']
    gdrive_client_id = your-client-id.apps.googleusercontent.com
    gdrive_client_secret = your-client-secret

>Then pull the data:

```bash
dvc pull
```

>To track and push new or updated data:

```bash
dvc add data/
git commit -am "Add/update data"
git push
dvc push
```

>Secrets like your OAuth client ID/secret are stored only in .dvc/config.local, which is automatically Git-ignored.

>.dvc/config contains the remote URL and is safe to commit.

>GitHub will block pushes if secrets are accidentally committed.

>DVC uses .dvc/cache/ to store data versions — no need to manage this manually.

## Access to gmail via api

>Copy credentials.json into the project root from your secure backup (e.g. USB)

## Run app 

```bash
python main.py
```