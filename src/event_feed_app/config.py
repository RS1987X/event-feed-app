# src/event_feed_app/config.py
from dataclasses import dataclass, asdict, field
import os
from typing import Optional, Dict, Any

try:
    # optional; if present we load a .env automatically
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(override=False)
except Exception:
    pass

DEFAULT_LANG_WHITELIST = "en,sv,fi,da,no,de,fr,is"

def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

@dataclass(frozen=True)
class Settings:
    # -------- General ----------
    taxonomy_version: str = os.getenv("TAXONOMY_VERSION", "v3")
    lang_whitelist: str   = os.getenv("LANG_WHITELIST", DEFAULT_LANG_WHITELIST)
    run_metrics_csv: str  = os.getenv("RUN_METRICS_CSV", "outputs/runs_metrics_v2.csv")
    log_level: str        = os.getenv("LOG_LEVEL", "INFO")

    # -------- IO ---------------
    gcs_silver_root: str  = os.getenv("GCS_SILVER_ROOT", "event-feed-app-data/silver_normalized/table=press_releases")
    output_local_csv: str = os.getenv("OUTPUT_LOCAL_CSV", "outputs/assignments.csv")
    output_gcs_uri: str   = os.getenv("OUTPUT_GCS_URI", "")
    eval_output_dir: str  = os.getenv("EVAL_OUTPUT_DIR", "outputs/eval")

    # -------- Models -----------
    emb_model_name: str   = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")
    e5_prefix: bool       = bool(int(os.getenv("E5_PREFIX", "1")))
    emb_batch_size: int   = int(os.getenv("EMB_BATCH_SIZE", "32"))
    max_seq_tokens: int   = int(os.getenv("MAX_SEQ_TOKENS", "384"))

    # -------- Fusion/temps -----
    fuse_alpha: float     = float(os.getenv("FUSE_ALPHA", "0.6"))
    temp_emb: float       = float(os.getenv("TEMP_EMB", "0.6"))
    temp_lsa: float       = float(os.getenv("TEMP_LSA", "0.8"))

    # -------- Thresholds -------
    cat_assign_min_sim: float = float(os.getenv("CAT_ASSIGN_MIN_SIM", "0.055"))
    cat_low_margin: float     = float(os.getenv("CAT_LOW_MARGIN", "0.001"))
    rule_conf_thr: float      = float(os.getenv("RULE_CONF_THR", "0.75"))

    # -------- Subset/knn -------
    use_subset: bool      = bool(int(os.getenv("USE_SUBSET", "0")))
    subset_clusters: int  = int(os.getenv("SUBSET_CLUSTERS", "100"))
    subset_per_cluster: int= int(os.getenv("SUBSET_PER_CLUSTER", "6"))
    knn_k: int            = int(os.getenv("KNN_K", "10"))

    # -------- Text -------------
    snippet_chars: int    = int(os.getenv("SNIPPET_CHARS", "600"))

    #------------Abstain other--------
    abstain_enable: bool = _env_bool("ABSTAIN_ENABLE", True)
    abstain_other_cid: str = os.getenv("ABSTAIN_OTHER_CID", "other_corporate_update")
    abstain_min_prob: float = float(os.getenv("ABSTAIN_MIN_PROB", "0.08"))
    abstain_min_margin: float = float(os.getenv("ABSTAIN_MIN_MARGIN", "0.03"))
    abstain_min_lang_conf: float = float(os.getenv("ABSTAIN_MIN_LANG_CONF", "0.40"))
    abstain_require_agreement: bool = _env_bool("ABSTAIN_REQUIRE_AGREEMENT", False)


    # helper: convert to dict (useful for logging)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_overrides(**kwargs) -> "Settings":
        """
        Build Settings and apply runtime overrides (e.g., parsed CLI flags).
        Only keys that match fields will be overridden.
        """
        base = Settings()
        current = base.to_dict()
        current.update({k: v for k, v in kwargs.items() if k in current and v is not None})
        # rebuild frozen dataclass with updates
        return Settings(**current)  # type: ignore[arg-type]
