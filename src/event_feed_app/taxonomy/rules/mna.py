# event_taxonomies/rules/mna.py
from __future__ import annotations
import re
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, any_present, any_present_morph, any_present_inflected
from .constants import (
    RULE_CONF_BASE,
    CURRENCY_AMOUNT,
    PER_SHARE_RX,
    MNA_FIXED_PHRASES,
    MNA_ACTION_STEMS_EN,
    MNA_ACTION_STEMS_NORDIC,
    MNA_OBJECT_NOUNS,
    MNA_CONTEXT_TERMS, BUYBACK_EXCLUDE, MNA_CORE
)
from event_feed_app.models.spacy_ner import ner_tag_sentences

# Restricted suffix sets (local to rule)
EN_VERB_SUFFIX = r"(?:|s|ed|ing)"
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"

def _hits(sentence: str):
    """Collect action/object/context cues for a single sentence."""
    action = (
        any_present(sentence, MNA_FIXED_PHRASES) +
        any_present_inflected(sentence, MNA_ACTION_STEMS_EN, EN_VERB_SUFFIX) +
        any_present_inflected(sentence, MNA_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX)
    )
    obj = any_present_morph(sentence, MNA_OBJECT_NOUNS)
    ctx = []
    if CURRENCY_AMOUNT.search(sentence):
        ctx.append("currency_amount")
    ctx += any_present_morph(sentence, MNA_CONTEXT_TERMS)
    if PER_SHARE_RX.search(sentence):
        ctx.append("per_share")
    strong = bool(any_present(sentence, MNA_FIXED_PHRASES))
    return action, obj, ctx, strong

@register
class MnaHighSignalRule:
    """
    High-precision transactional M&A:
    - Cheap prefilter (action + object)
    - Sentence-level NER (ORG) gating
    - Same-sentence or adjacent-sentence backoff
    - Locks when it fires
    """
    name = "mna"   # must match constants key
    priority = -32  # before generic backstops

    def run(self, ctx: Ctx, keywords):
        title, body = ctx.title or "", ctx.body or ""
        text = f"{title} {body}"

        # Cheap prefilter: avoid NER if clearly not M&A-ish
        cheap_action = (
            any_present(text, MNA_FIXED_PHRASES) or
            any_present_inflected(text, MNA_ACTION_STEMS_EN, EN_VERB_SUFFIX) or
            any_present_inflected(text, MNA_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX)
        )
        cheap_object = any_present_morph(text, MNA_OBJECT_NOUNS)
        if not (cheap_object and cheap_action):
            return False, None, Details(self.name, {}, 0.0, False)

        sents = ner_tag_sentences(text)

        # Pass 1: same-sentence requirement (with ORG gating)
        for sent in sents:
            s = sent["text"]
            orgs = [e for e in sent["entities"] if e["type"] == "ORG"]
            if not orgs:
                continue
            a, o, c, strong = _hits(s)
            if not (o and (a or strong) and (c or strong)):
                continue

            # 2 ORGs for acquisitions/mergers, 1 OK for tenders/spin/sale titles
            needs_two = any_present_morph(s, ["acquisition", "merger", "business combination", "combination"])
            if len(orgs) < (2 if needs_two else 1):
                continue

            det = list(dict.fromkeys(a + o + c))
            return True, "mna", Details(self.name, {"mna": det}, RULE_CONF_BASE.get(self.name, 0.90), lock=True)

        # Pass 2: adjacent sentences backoff (s_i + s_{i+1})
        for i in range(len(sents) - 1):
            s1, s2 = sents[i]["text"], sents[i + 1]["text"]
            orgs = [e for e in sents[i]["entities"] + sents[i + 1]["entities"] if e["type"] == "ORG"]
            if not orgs:
                continue
            a1, o1, c1, strong1 = _hits(s1)
            a2, o2, c2, strong2 = _hits(s2)
            a, o, c = a1 + a2, o1 + o2, c1 + c2
            strong = strong1 or strong2
            if o and (a or strong) and (c or strong) and len(orgs) >= 2:
                det = list(dict.fromkeys(a + o + c))
                return True, "mna", Details(self.name, {"mna": det}, RULE_CONF_BASE.get(self.name, 0.90), lock=True)

        return False, None, Details(self.name, {}, 0.0, False)

@register
class MnaCoreBackstopRule:
    name = "mna"     # ← same label
    priority = -28               # after the high-signal rule

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title or ''} {ctx.body or ''}"
        if MNA_CORE.search(text) and not BUYBACK_EXCLUDE.search(text):
            conf = max(0.0, RULE_CONF_BASE[self.name] - 0.03)
            return True, "mna", Details(
                rule=self.name,                          # ← same rule label
                hits={"mna": ["mna_core"]},
                rule_conf=conf,
                lock=False,
                notes="source=regex_backstop"            # optional
            )
        return False, None, Details(self.name, {}, 0.0, False)