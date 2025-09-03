import langid
import pandas as pd
import numpy as np
def setup_langid(whitelist: str):
    
    if whitelist.strip():
        langid.set_languages([s.strip() for s in whitelist.split(",") if s.strip()])
    return langid

def detect_language_batch(titles: pd.Series, bodies: pd.Series, max_len: int = 1200, langid=None):
    langs, confs = [], []
    for t, b in zip(titles.fillna(""), bodies.fillna("")):
        txt = (str(t).strip() + " " + str(b).strip()[:max_len]).strip()
        if not txt:
            langs.append("und"); confs.append(0.0); continue
        lang, score = langid.classify(txt)
        try:
            conf = 1.0 - float(np.exp(-float(score) / 10.0))
        except Exception:
            conf = 0.0
        langs.append(lang); confs.append(round(conf, 4))
    return langs, confs


# normalize language codes to the ones you actually have files for
def normalize_lang(code: str) -> str:
    code = (code or "und").lower()
    if code.startswith("sv"): return "sv"
    if code.startswith("fi"): return "fi"
    if code.startswith("da"): return "da"
    # keep 'en' for English; everything else returns as-is and will fall back to English
    if code.startswith("en"): return "en"
    return code
