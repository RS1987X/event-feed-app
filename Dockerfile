# Dockerfile

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# 1. Install OS-level build tools for C-extensions, Spacy, DVC, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      libffi-dev \
      libssl-dev \
      python3-dev \ 
      curl \
      git &&\
      #ca-certificates && \
    #update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy & install Python dependencies
COPY . .
 #.env requirements.txt . /


RUN pip install --upgrade pip \
    && pip install --no-cache-dir "dvc[gdrive]" \
    && pip install --no-cache-dir -r requirements.txt

# 3. Download SpaCy language models at build time
#    (so they’re baked into the image, rather than downloaded at runtime)
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download xx_ent_wiki_sm

# 4. Copy your application code
#COPY . .

# 3) make sure SERVICE_ACCOUNT_JSON is set (fallback to sa-key.json)
#ENV SERVICE_ACCOUNT_JSON=${SERVICE_ACCOUNT_JSON:-sa-key.json}

# 5. On container start: pull the latest DVC data, then run your app
#ENTRYPOINT ["sh", "-c", "dvc pull && python main.py"]
# ENTRYPOINT ["sh", "-c", "dvc pull --force \
#             && python main_cloud.py"]

# ENTRYPOINT ["sh", "-c", "\
#   # inject the env-var into DVC's local config, then pull, then run \
#   dvc remote modify mygdrive --local \
#      gdrive_service_account_json_file_path $SERVICE_ACCOUNT_JSON && \
#   dvc pull --force && \
#   python main_cloud.py"]

# ENTRYPOINT ["sh","-c", "\
#   dvc pull -r mygdrive && \
#   python main_cloud.py \
# "]

# ENTRYPOINT ["sh","-c", "\
#   dvc pull -r mygdrive && \
#   python main_cloud.py --port ${PORT:-8080}"]


#ENTRYPOINT ["sh","-c","echo \"Starting with PORT=$PORT\" && python main_cloud.py"]

ENTRYPOINT ["sh","-c","echo \"Starting with PORT=$PORT\" && exec python -u main_cloud.py"]
#ENTRYPOINT ["python", "main_cloud.py"]