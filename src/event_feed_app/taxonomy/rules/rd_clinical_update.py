from __future__ import annotations
from typing import Dict, Any, Tuple
from .registry import register, Details
from .context import Ctx, norm, any_present_morph, proximity_cooccurs_morph
from .constants import (
    RULE_CONF_BASE,REPORT_TITLE_TERMS, MAJOR_HOLDINGS_VETOS, ORDER_CONTRACT_HEAD_TERMS,
    DISSEMINATION_OR_RESULTS, REGULATORY_VETO,
    CLINICAL_ANCHORS, CONGRESS_TERMS, BIOMED_CONTEXT,)

def _has_any(text: str, terms: Tuple[str, ...], stem: int = 3) -> bool:
    return bool(any_present_morph(text, list(terms), min_stem=stem))

def _title_report_like(title_norm: str) -> bool:
    return _has_any(title_norm, REPORT_TITLE_TERMS, stem=3)

def _is_major_holdings(text_norm: str) -> bool:
    return _has_any(text_norm, MAJOR_HOLDINGS_VETOS, stem=4)

def _orderish_title_blocks_without_dissemination(title_norm: str, text_norm: str) -> bool:
    if not _has_any(title_norm, ORDER_CONTRACT_HEAD_TERMS, stem=4):
        return False
    return not _has_any(text_norm, DISSEMINATION_OR_RESULTS, stem=3)

def _title_present_at_congress(title_original: str) -> bool:
    # title-level “present/abstract/poster/oral” near a congress word/acronym
    near, _ = proximity_cooccurs_morph(
        title_original,
        list(DISSEMINATION_OR_RESULTS),
        list(CONGRESS_TERMS),
        window=8, min_stem=3
    )
    return near

@register
class HiSigRdClinicalUpdateV3:
    """
    High-precision R&D / Clinical & Scientific update.

    Path A (core): clinical/scientific ANCHOR + DISSEMINATION within proximity.
    Path B (congress announce): DISSEMINATION near CONGRESS term (+ biomed context),
                                so “will present data at [ASCO/ESMO/ESC…]” is captured.
    Structural vetoes: report titles, TR-1/major holdings, and order-ish titles without dissemination.
    """
    name = "hi_sig_rd_clinical_update_v3"
    priority = 120

    def run(self, ctx: Ctx, keywords: Dict[str, Any]):
        title_n = norm(ctx.title)
        text_n  = ctx.text_norm

        # ---- VETOES
        if _title_report_like(title_n):
            return False, None, None
        if _is_major_holdings(text_n):
            return False, None, None
        if _orderish_title_blocks_without_dissemination(title_n, text_n):
            return False, None, None
        if any_present_morph(ctx.text_original, list(REGULATORY_VETO), min_stem=4):
            # safest: abstain so the legal/regulatory rule or keywords pick it up
            return False, None, None
        
        # ---- Path A: anchor + dissemination + proximity (body-level)
        has_anchor = _has_any(text_n, CLINICAL_ANCHORS, stem=3)
        has_disse  = _has_any(text_n, DISSEMINATION_OR_RESULTS, stem=3)
        if has_anchor and has_disse:
            nearA, _ = proximity_cooccurs_morph(
                ctx.text_original,
                list(CLINICAL_ANCHORS),
                list(DISSEMINATION_OR_RESULTS),
                window=25, min_stem=3
            )
            if nearA:
                det = Details(
                    rule=self.name,
                    hits={"rd_clinical_update": ["anchor+dissemination+proximity"]},
                    rule_conf=0.83,
                    lock=False,
                )
                return True, "rd_clinical_update", det

        # ---- Path B: congress announcement (title or body), with biomed context
        has_congress = _has_any(text_n, CONGRESS_TERMS, stem=3) or _title_present_at_congress(ctx.title)
        has_biomed   = has_anchor or _has_any(text_n, BIOMED_CONTEXT, stem=3)

        if has_congress and has_disse and has_biomed:
            # Prefer proximity in body if anchor is missing; allow tighter window
            nearB_title = _title_present_at_congress(ctx.title)
            nearB_body, _ = proximity_cooccurs_morph(
                ctx.text_original,
                list(DISSEMINATION_OR_RESULTS),
                list(CONGRESS_TERMS),
                window=15, min_stem=3
            )
            if nearB_title or nearB_body:
                det = Details(
                    rule=self.name,
                    hits={"rd_clinical_update": ["congress_announce"]},
                    rule_conf=0.81,   # slightly lower than Path A but above threshold
                    lock=False,
                )
                return True, "rd_clinical_update", det

        return False, None, None
