# Use a slim Python base
FROM python:3.11-slim

# System deps (Polars needs some)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-min.txt .
RUN pip install --no-cache-dir -r requirements-min.txt

# Copy your code
COPY . .

# Default command: run the single-pass script
# (Replace with your actual filename if different)
CMD ["python", "gcr_job_main.py"]
