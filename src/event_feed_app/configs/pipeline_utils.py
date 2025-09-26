# event_feed_app/configs/pipeline/utils.py
import json, os
from event_feed_app.config import Settings

TUNABLE_KEYS = {
    "fuse_alpha", "temp_emb", "temp_lsa",
    "cat_assign_min_sim", "cat_low_margin",
    "rule_conf_thr",
    "abstain_enable", "abstain_other_cid", "abstain_min_prob",
    "abstain_min_margin", "abstain_min_lang_conf", "abstain_require_agreement",
}

def load_pipeline_snapshot(cfg: Settings, path: str | None = None, respect_env: bool = True) -> Settings:
    path = path or cfg.pipeline_config_json
    if not path or not os.path.exists(path):
        return cfg
    with open(path, "r") as f:
        snap = json.load(f)

    # keep only keys that exist on Settings & are tunable
    snap = {k: v for k, v in snap.items() if k in TUNABLE_KEYS and hasattr(cfg, k)}

    # let explicit env vars win (optional)
    if respect_env:
        snap = {k: v for k, v in snap.items() if os.getenv(k.upper()) is None}

    return Settings.from_overrides(**snap)
