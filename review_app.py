# review_app.py
# streamlit run review_app.py
import os, json, re, datetime
import pandas as pd
import streamlit as st
import gcsfs

ARTIFACT_ROOT = os.getenv("ARTIFACT_ROOT", "gs://event-feed-app-data/unsup_clusters")

st.set_page_config(page_title="PR Category Review", layout="wide")
st.title("Press Release Category Review")

fs = gcsfs.GCSFileSystem()

def list_runs(prefix: str) -> list[str]:
    # returns list of gs://.../<timestamp> directories
    # gcsfs.ls gives full paths for objects; we filter “folders”
    paths = fs.ls(prefix)
    runs = sorted({p for p in paths if re.search(r"/\d{4}-\d{2}-\d{2}T\d{6}Z/?$", p)})
    return runs

def load_json(path: str) -> dict:
    with fs.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def load_csv(path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        return pd.read_csv(f)

def save_csv(df: pd.DataFrame, path: str):
    # atomic-ish write
    with fs.open(path, "wb") as f:
        df.to_csv(f, index=False)

# --- Pick run ---
runs = list_runs(ARTIFACT_ROOT)
if not runs:
    st.error(f"No runs found under {ARTIFACT_ROOT}")
    st.stop()

run_choice = st.sidebar.selectbox("Run folder", runs, index=len(runs)-1)
run_dir = run_choice.rstrip("/")

categories_path = f"{run_dir}/categories.json"
review_pack_path = f"{run_dir}/review_samples.csv"
assign_path      = f"{run_dir}/assignments_with_categories.csv"
decisions_path   = f"{run_dir}/review_decisions.csv"

# --- Load categories and make dropdown options ---
cats = load_json(categories_path).get("categories", [])
cid_to_name = {c["cid"]: c["label"] for c in cats if c.get("active", True)}
name_to_cid = {v: k for k, v in cid_to_name.items()}
cat_options = list(cid_to_name.values())  # names shown in dropdown

st.sidebar.markdown("### Categories")
st.sidebar.write(pd.DataFrame([{"cid": k, "label": v} for k, v in cid_to_name.items()]))


# =======================
# INIT SESSION STORAGE  ✅  <-- PUT THIS BLOCK HERE
# =======================
if "pending" not in st.session_state:
    st.session_state.pending = []  # a list of dicts

def upsert_pending(rec: dict):
    """Insert or replace a pending decision by (press_release_id, run_ts)."""
    key = (rec.get("press_release_id",""), rec.get("run_ts",""))
    rows = st.session_state.pending
    for i, r in enumerate(rows):
        if (r.get("press_release_id",""), r.get("run_ts","")) == key:
            rows[i] = rec
            return
    rows.append(rec)

# --- Load review rows (prefer review_samples; fallback to assignments) ---
try:
    df = load_csv(review_pack_path)
    # expected columns: cluster,cid,category_name,title,snippet,_silhouette,cat_sim,...
    if "category_name" not in df.columns and "cid" in df.columns:
        df["category_name"] = df["cid"].map(cid_to_name).fillna(df.get("category_name",""))
except FileNotFoundError:
    df = load_csv(assign_path)
    # build minimal snippet from what we have (assignments doesn't include body)
    title = df.get("title", "")
    df["title"] = title if isinstance(title, pd.Series) else ""
    df["snippet"] = df.get("source_url", "").fillna("").astype(str)  # placeholder
    df["category_name"] = df["cid"].map(cid_to_name).fillna(df.get("category_name",""))

# --- Uncertainty sorting: lowest sim first, then lowest silhouette ---
def uncertainty_row(r):
    sim = float(r.get("cat_sim", r.get("cat_similarity", 0.0)) or 0.0)
    sil = float(r.get("_silhouette", 0.0) or 0.0)
    return (sim, sil)

df["_sim_"] = pd.to_numeric(df.get("cat_sim", 0.0), errors="coerce").fillna(0.0)
df["_sil_"] = pd.to_numeric(df.get("_silhouette", 0.0), errors="coerce").fillna(0.0)
df = df.sort_values(by=["_sim_", "_sil_"], ascending=[True, True]).reset_index(drop=True)

st.caption(f"Run: `{run_dir}` — rows: {len(df):,}")

# --- Batch size & simple filters ---
n_show = st.sidebar.slider("How many to review now", 10, 200, 40, step=10)
only_decisions = st.sidebar.multiselect("Focus decisions", ["needs_review_low_sim","needs_review_margin","auto_assign"], default=["needs_review_low_sim","needs_review_margin"])
if "decision" in df.columns and only_decisions:
    df = df[df["decision"].isin(only_decisions)].reset_index(drop=True)

view = df.head(n_show).copy()

# --- Collect edits ---
st.subheader("Review queue (uncertain first)")

for i, row in view.iterrows():
    with st.expander(f"#{i+1} — {row.get('company_name','')} — {row.get('title','')[:120]}"):
        cols = st.columns([3,2,2])
        with cols[0]:
            st.write("**Title**:", row.get("title",""))
            st.write(row.get("snippet",""))
            st.caption(f"press_release_id: {row.get('press_release_id','')}")
        with cols[1]:
            st.metric("Category sim", f"{row.get('cat_sim',0):.3f}")
            st.metric("Silhouette", f"{row.get('_silhouette',0):.3f}")
            st.write("Current:", row.get("category_name", ""))
        with cols[2]:
            default_name = row.get("category_name","")
            try:
                idx = cat_options.index(default_name)
            except ValueError:
                idx = 0
            new_name = st.selectbox("Assign to category", cat_options, index=idx, key=f"cat_{i}")
            reason   = st.text_input("Notes (optional)", "", key=f"note_{i}")
            if st.button("Save decision", key=f"save_{i}"):
                rec = {
                    "press_release_id": row.get("press_release_id",""),
                    "old_cid": name_to_cid.get(default_name, ""),
                    "old_name": default_name,
                    "new_cid": name_to_cid.get(new_name, ""),
                    "new_name": new_name,
                    "cat_sim": float(row.get("cat_sim", 0.0) or 0.0),
                    "_silhouette": float(row.get("_silhouette", 0.0) or 0.0),
                    "run_ts": run_dir.rsplit("/",1)[-1],
                    "ts_recorded": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "notes": reason,
                }
                upsert_pending(rec)
                st.success("Queued. It’s in the Pending decisions table below.")

st.divider()
st.subheader(f"Pending decisions ({len(st.session_state.pending)})")

pend_df = pd.DataFrame(st.session_state.pending)
if not pend_df.empty:
    st.dataframe(pend_df, use_container_width=True)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Clear pending"):
            st.session_state.pending = []
            st.experimental_rerun()
    with c2:
        if st.button("Write decisions to GCS"):
            try:
                # load existing decisions (if any)
                try:
                    existing = load_csv(decisions_path)
                except FileNotFoundError:
                    existing = pd.DataFrame()
                all_df = pd.concat([existing, pend_df], ignore_index=True)

                # de-dup by (press_release_id, run_ts) — keep last
                if not all_df.empty and "press_release_id" in all_df.columns:
                    all_df = (all_df
                              .drop_duplicates(subset=["press_release_id","run_ts"], keep="last")
                              .reset_index(drop=True))
                save_csv(all_df, decisions_path)
                st.session_state.pending = []
                st.success(f"Wrote {len(pend_df)} decisions → {decisions_path}")
            except Exception as e:
                st.error(f"Failed to write decisions: {e}")
else:
    st.caption("No pending decisions yet. Save a few above, then write them to GCS.")


st.caption("Tip: run `streamlit run review_app.py` (not `python review_app.py`) to avoid SessionState warnings.")
