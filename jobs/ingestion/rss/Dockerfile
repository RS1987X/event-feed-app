# Small, fast base
FROM python:3.11-slim

# Keep Python output unbuffered, no .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System certs; no compiler needed for these wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements_rss_job.txt .
RUN pip install --no-cache-dir -r requirements_rss_job.txt

# Copy your job script
# (Assumes the file is named rss_single_feed_job.py in your build context)
COPY gcr_job_rss.py .

# Run as non-root (optional but good practice)
RUN useradd -m app && chown -R app:app /app
USER app

# Default command: run the RSS job once
CMD ["python", "gcr_job_rss.py"]
