#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Analyze Gmail press release publication timing.

Two modes:
  1) --mode local-labels: Parse timestamps from labels.csv (fallback, quick)
  2) --mode gcs: Read from GCS Parquet silver using published_utc (preferred)

Outputs written under outputs/analysis/<run_tag>/
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import pytz

# Local helper import (works within repo)
try:
    from event_feed_app.utils.gcs_io import load_window_parquet_df
except Exception:
    load_window_parquet_df = None  # type: ignore

STHLM = pytz.timezone("Europe/Stockholm")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def window_mask(hh: int, mm: int, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    t = hh * 60 + mm
    a = start_h * 60 + start_m
    b = end_h * 60 + end_m
    if a <= b:
        return a <= t < b
    return (t >= a) or (t < b)  # wrap midnight


def summarize_times(df: pd.DataFrame, *, dt_col: str) -> Dict[str, Any]:
    df = df.copy()
    # Normalize to Stockholm
    s = df[dt_col]
    # parse
    s = pd.to_datetime(s, utc=True, errors="coerce")
    # Non-UTC fallback (assume already local naive):
    # Keep only parsed
    df["dt"] = s
    df = df.dropna(subset=["dt"]).copy()
    if df.empty:
        return {
            "N": 0,
            "windows": {},
            "hourly": pd.Series(dtype=int),
            "dow": pd.Series(dtype=int),
            "df": df,
        }
    df["dt_swe"] = df["dt"].dt.tz_convert("Europe/Stockholm")
    df["hour"] = df["dt_swe"].dt.hour
    df["minute"] = df["dt_swe"].dt.minute

    # Windows
    def w(name, a_h, a_m, b_h, b_m):
        m = df.apply(lambda r: window_mask(int(r["hour"]), int(r["minute"]), a_h, a_m, b_h, b_m), axis=1)
        return int(m.sum())

    counts = {
        "evening_1730_2200": w("evening", 17, 30, 22, 0),
        "late_night_2200_0600": w("late", 22, 0, 6, 0),
        "early_0600_0730": w("early", 6, 0, 7, 30),
        "premarket_0730_0900": w("premarket", 7, 30, 9, 0),
        "market_0900_1730": w("market", 9, 0, 17, 30),
    }

    N = len(df)
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    # 30-minute buckets (0..47)
    df["mins"] = df["dt_swe"].dt.hour * 60 + df["dt_swe"].dt.minute
    df["hh_bucket"] = (df["mins"] // 30).astype(int)

    def _hh_label(bucket: int) -> str:
        start = int(bucket) * 30
        end = (start + 30) % 1440
        sh, sm = divmod(start, 60)
        eh, em = divmod(end, 60)
        return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"

    labels = [_hh_label(b) for b in range(48)]
    half_hourly = (
        df["hh_bucket"].map(lambda b: labels[int(b)]).value_counts().reindex(labels, fill_value=0)
    )
    dow = df.groupby(df["dt_swe"].dt.day_name()).size()

    return {"N": N, "windows": counts, "hourly": hourly, "half_hourly": half_hourly, "dow": dow, "df": df}


def run_local_labels(out_dir: Path) -> None:
    labels_path = Path("labels.csv")
    if not labels_path.exists():
        raise SystemExit("labels.csv not found in repo root")
    labels = pd.read_csv(labels_path)

    # Extract timestamps from text fields (snippet / title)
    import re

    patterns = [
        r"(?:Published|Publicerad|Publicerat|Publicerades|Julkaistu|Udgivet|Publisert)\s*:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s*(CEST|CET|EEST|EET)?",
        r"(?:Published on|Publicerad den)\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s*(CEST|CET|EEST|EET)?",
    ]

    def extract_ts(row) -> Optional[datetime]:
        for col in ("snippet", "title"):
            val = str(row.get(col, ""))
            if not val or val == "nan":
                continue
            for pat in patterns:
                m = re.search(pat, val, flags=re.IGNORECASE)
                if m:
                    date_str, hh, mm, ss, tz = m.groups()
                    ss = ss or "00"
                    tz = (tz or "CEST").upper()
                    dt_local = datetime.strptime(f"{date_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
                    tz_map = {
                        "CET": pytz.timezone("Europe/Stockholm"),
                        "CEST": pytz.timezone("Europe/Stockholm"),
                        "EET": pytz.timezone("Europe/Helsinki"),
                        "EEST": pytz.timezone("Europe/Helsinki"),
                    }
                    loc = tz_map.get(tz, pytz.timezone("Europe/Stockholm"))
                    dt_loc_aware = loc.localize(dt_local, is_dst=None)
                    return dt_loc_aware.astimezone(STHLM)
        return None

    labels["dt_swe"] = labels.apply(extract_ts, axis=1)  # type: ignore[call-overload]
    ok = labels.dropna(subset=["dt_swe"]).copy()
    ok["dt"] = ok["dt_swe"].dt.tz_convert("UTC")

    res = summarize_times(ok, dt_col="dt")

    # Write outputs
    _ensure_dir(out_dir)
    ok[["press_release_id", "company_name", "title", "dt_swe"]].to_csv(out_dir / "gmail_publication_times_local.csv", index=False)
    res["hourly"].to_csv(out_dir / "gmail_hourly_counts_local.csv", header=["count"])
    res["half_hourly"].to_csv(out_dir / "gmail_30min_counts_local.csv", header=["count"])
    pd.Series(res["windows"]).to_csv(out_dir / "gmail_window_counts_local.csv", header=["count"])

    print(f"Local labels summary (N={res['N']}):")
    for k, v in res["windows"].items():
        pct = (v / res["N"] * 100.0) if res["N"] else 0.0
        print(f"  {k:>22}: {v:5d} ({pct:4.1f}%)")


def run_gcs(out_dir: Path, gcs_path: str, start_utc: datetime, end_utc: datetime) -> None:
    if load_window_parquet_df is None:
        raise SystemExit("gcs_io helpers not available; ensure you installed deps and are in repo root.")

    # Try to ensure source filter; if not included, append partition
    # This respects layouts like gs://bucket/silver/source=gmail/date=YYYY-MM-DD/...
    base = gcs_path.rstrip("/")
    if "/source=" not in base:
        base = f"{base}/source=gmail"

    print(f"Reading GCS parquet from {base} (window: {start_utc.isoformat()} → {end_utc.isoformat()}) …")
    df = load_window_parquet_df(base, start_utc, end_utc)
    if df.empty:
        print("No rows returned from GCS in this window.")
        return

    # Ensure published_utc exists and is parseable (may actually be ingestion time for gmail)
    if "published_utc" not in df.columns:
        if "ingested_at" in df.columns:
            df["published_utc"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")
        else:
            raise SystemExit("Input has no published_utc or ingested_at columns; adjust loader.")

    # Try to recover true publication time from the body/full_text for Gmail sources.
    # Many pre-market releases arrive earlier than the ingestion job and would otherwise
    # be incorrectly clustered at ~09:00 local if we only use ingestion timestamps.
    import re
    patterns = [
        r"(?:Published|Publicerad|Publicerat|Publicerades|Julkaistu|Udgivet|Publisert)\s*:\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s*(CEST|CET|EEST|EET)?",
        r"(?:Published on|Publicerad den)\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s*(CEST|CET|EEST|EET)?",
    ]

    tz_map = {
        "CET": pytz.timezone("Europe/Stockholm"),
        "CEST": pytz.timezone("Europe/Stockholm"),
        "EET": pytz.timezone("Europe/Helsinki"),
        "EEST": pytz.timezone("Europe/Helsinki"),
    }

    def parse_body_ts(txt: str) -> Optional[datetime]:
        if not txt:
            return None
        s = str(txt)
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                date_str, hh, mm, ss, tz = m.groups()
                ss = ss or "00"
                tz = (tz or "CEST").upper()
                try:
                    dt_local = datetime.strptime(f"{date_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
                loc = tz_map.get(tz, pytz.timezone("Europe/Stockholm"))
                aware = loc.localize(dt_local, is_dst=None)
                return aware.astimezone(timezone.utc)
        return None

    # Attempt parsing from full_text / body-like columns
    candidate_cols = [c for c in ["full_text", "body", "raw_body"] if c in df.columns]
    parsed_utcs: list[Optional[datetime]] = []
    if candidate_cols:
        for _, row in df.iterrows():
            ts: Optional[datetime] = None
            for c in candidate_cols:
                ts = parse_body_ts(row.get(c, ""))
                if ts:
                    break
            parsed_utcs.append(ts)
        df["parsed_published_utc"] = parsed_utcs
    else:
        df["parsed_published_utc"] = None

    N_parsed = int(df["parsed_published_utc"].notna().sum())
    if N_parsed:
        print(f"Recovered explicit 'Published:' timestamps for {N_parsed} / {len(df)} rows ({N_parsed/len(df)*100:.1f}%).")
    else:
        print("No explicit 'Published:' timestamps recovered; using published_utc values as-is.")

    # Effective publication time: prefer parsed, fallback to published_utc
    eff = df["parsed_published_utc"].where(df["parsed_published_utc"].notna(), df["published_utc"])
    df["effective_published_utc"] = pd.to_datetime(eff, utc=True, errors="coerce")

    # Summarize using effective publication time
    res = summarize_times(df, dt_col="effective_published_utc")

    _ensure_dir(out_dir)
    # Save per-row reduced output to keep file small
    # Important: align all columns to the filtered/resolved rows used in `res["df"]`
    idx = res["df"].index
    cols_dict: Dict[str, Any] = {
        "effective_published_utc": res["df"]["dt"].dt.tz_convert("UTC"),
        "effective_published_swe": res["df"]["dt_swe"],
    }
    for c in ("press_release_id", "company_name", "title"):
        if c in df.columns:
            cols_dict[c] = df.loc[idx, c]
    if "published_utc" in df.columns:
        cols_dict["raw_published_utc"] = df.loc[idx, "published_utc"]
    if "parsed_published_utc" in df.columns:
        cols_dict["parsed_published_utc"] = df.loc[idx, "parsed_published_utc"]

    slim = pd.DataFrame(cols_dict)
    slim.to_csv(out_dir / "gmail_publication_times_gcs.csv", index=False)
    res["hourly"].to_csv(out_dir / "gmail_hourly_counts_gcs.csv", header=["count"])
    res["half_hourly"].to_csv(out_dir / "gmail_30min_counts_gcs.csv", header=["count"])
    pd.Series(res["windows"]).to_csv(out_dir / "gmail_window_counts_gcs.csv", header=["count"])

    print(f"GCS summary (N={res['N']} effective timestamps):")
    for k, v in res["windows"].items():
        pct = (v / res["N"] * 100.0) if res["N"] else 0.0
        print(f"  {k:>22}: {v:5d} ({pct:4.1f}%)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Gmail press release publication timing.")
    ap.add_argument("--mode", choices=["local-labels", "gcs"], default="local-labels",
                    help="Data source: local labels.csv parser or GCS parquet (preferred)")
    ap.add_argument("--gcs-path", default=os.getenv("GCS_SILVER_ROOT", ""),
                    help="GCS silver root (e.g., gs://<bucket>/silver). If missing source partition, '/source=gmail' will be appended.")
    ap.add_argument("--days", type=int, default=180, help="Lookback window in days for GCS mode.")
    ap.add_argument("--start-utc", type=str, default=None, help="ISO8601 UTC start (overrides --days)")
    ap.add_argument("--end-utc", type=str, default=None, help="ISO8601 UTC end (default: now)")
    ap.add_argument("--tag", type=str, default=None, help="Run tag for output dir name")
    return ap.parse_args()


def parse_iso8601_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def main() -> None:
    args = parse_args()

    end_utc = parse_iso8601_utc(args.end_utc) or _now_utc()
    start_utc = parse_iso8601_utc(args.start_utc) or (end_utc - timedelta(days=args.days))

    tag = args.tag or datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path("outputs/analysis") / f"gmail_release_timing_{tag}"

    if args.mode == "local-labels":
        run_local_labels(out_dir)
    else:
        if not args.gcs_path:
            raise SystemExit("--gcs-path empty and $GCS_SILVER_ROOT not set.")
        run_gcs(out_dir, args.gcs_path, start_utc, end_utc)


if __name__ == "__main__":
    main()
