# Gmail Press Release Timing Analysis

This note documents when press releases arrive (Stockholm time) and what that means for scheduling the alerts pipeline.

## Data sources

- Quick local sample: `labels.csv` (parses "Published: … CEST/CET" in text)
  - Coverage: 2,755 rows total; 2,098 (76.2%) with extractable timestamps
- Full dataset (recommended): GCS "silver" Parquet with `published_utc`
  - Path: `gs://<your-bucket>/silver` (or `$GCS_SILVER_ROOT`) partitioned by `source`
  - Filter: `source=gmail`

Use the helper script to reproduce and extend the analysis:

```
# Local quick run (parses labels.csv)
python scripts/analyze_gmail_release_timing.py --mode local-labels

# GCS wide run (last 180 days by default)
python scripts/analyze_gmail_release_timing.py --mode gcs --gcs-path "$GCS_SILVER_ROOT"
```

Outputs will be saved under `outputs/analysis/gmail_release_timing_<timestamp>/`.

You will get:
- gmail_publication_times_*.csv — per-press release timestamps
- gmail_window_counts_*.csv — market window totals
- gmail_hourly_counts_*.csv — hour-of-day counts
- gmail_30min_counts_*.csv — 30-minute bucket counts (00:00–00:29, 00:30–00:59, …)

## Local findings (Gmail only, N=2,098 parsed)

- Evening 17:30–22:00: 16.1%
- Late night 22:00–06:00: 1.4%
- Early 06:00–07:30: 4.4%
- Pre-market 07:30–09:00: 30.6%
- Market hours 09:00–17:30: 47.5%

Hourly highlights:
- 07:00: 12.2%
- 08:00: 22.3%
- 17:00: 8.8%

Day-of-week: Thu/Fri/Mon are heaviest; weekends near zero.

Notes:
- Local results come from parsing timestamps embedded in the text ("Published: …"). They approximate the true publication time. The GCS-based analysis reads canonical `published_utc` and is preferred for decisions.

## Scheduling recommendations

Goal: Maximize actionable alerts before open, minimize runs, still capture after-close.

- Keep a final pre-open run at ~08:50 (captures the big 07:30–09:00 wave ≈ 30.6%).
- Keep the after-close run at ~17:45 (captures ≈ 16% same-evening publications quickly).
- The 07:30 run adds relatively little unique value (≈ 4.4%) compared to an 08:50 run; skip unless you need the extra ~1h20 earlier delivery.
- Optional mid-day run at 12:30 for intraday catch-up.

Example cron (Stockholm time):

```
# Alerts
50 8 * * 1-5   event-alerts-run --start-date $(date -d 'today' +\%F) --end-date $(date -d 'today' +\%F)
30 12 * * 1-5  event-alerts-run --start-date $(date -d 'today' +\%F) --end-date $(date -d 'today' +\%F)
45 17 * * 1-5  event-alerts-run --start-date $(date -d 'today' +\%F) --end-date $(date -d 'today' +\%F)
```

Important: Ensure Gmail ingestion finishes before the 08:50 run. Options:
- Shift ingestion earlier (e.g., 08:40), or
- Add two ingestions: ~08:40 and ~17:35 to feed the 17:45 alerts.

## Reproducing with GCS (preferred)

Run the script in `--mode gcs` for a longer lookback (e.g., 365 days) and validate the distribution with your full dataset:

```
python scripts/analyze_gmail_release_timing.py --mode gcs --gcs-path "$GCS_SILVER_ROOT" --days 365
```

This writes:
- gmail_publication_times_gcs.csv — per-press release timestamps (UTC and Stockholm)
- gmail_hourly_counts_gcs.csv — counts per hour
- gmail_window_counts_gcs.csv — window totals (evening, late night, early, premarket, market)

## Next steps

- Optionally, extend the script to include RSS and wires for a complete source mix.
- Add scheduled weekly job to regenerate the analysis and attach a small trend chart to Slack/Telegram.
- Adjust cron windows once the GCS-wide analysis confirms the local shape.
