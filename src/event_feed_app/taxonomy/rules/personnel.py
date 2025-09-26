from __future__ import annotations
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, tokenize, any_present, any_present_morph, any_present_inflected, proximity_cooccurs_morph
from .constants import RULE_CONF_BASE, PERSONNEL_ROLES, PERSONNEL_MOVE_STEMS_EN, PERSONNEL_MOVE_STEMS_NORDIC, PERSONNEL_FIXED_PHRASES, PERSONNEL_PROPOSAL_NOUNS, PERSONNEL_PROPOSAL_STEMS, AGM_TERMS
from event_feed_app.models.spacy_ner import detect_person_name_spacy

EN_VERB_SUFFIX = r"(?:|s|ed|ing)"
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"

EN_SUFFIXES = ("", "s", "ed", "ing")
SV_SUFFIXES = ("", "ar", "ade", "at", "er", "te", "t")

def _positions_inflected_single_token(toks, stems, suffixes):
    hits = []
    for i, tt in enumerate(toks):
        for s in stems:
            # exact inflection only (prevents join -> joint, assum -> assumption, depart -> department)
            if any(tt == (s + suf) for suf in suffixes):
                hits.append(i)
                break
    return hits

@register
class PersonnelRule:
    name = "role + move + name"
    priority = -35

    def run(self, ctx: Ctx, keywords):
        title, body = ctx.title or "", ctx.body or ""
        text = f"{title} {body}"
        original = f"{title}\n{body}"

        if (any_present_morph(text, PERSONNEL_PROPOSAL_NOUNS)
            and (any_present_inflected(text, PERSONNEL_PROPOSAL_STEMS, EN_VERB_SUFFIX)
                 or any_present_inflected(text, PERSONNEL_PROPOSAL_STEMS, SV_VERB_SUFFIX))
            and any_present_morph(text, AGM_TERMS)):
            return False, None, Details(self.name, {}, 0.0, False)

        role_hits = any_present_morph(text, PERSONNEL_ROLES)
        move_hits = (
            any_present(text, PERSONNEL_FIXED_PHRASES)
            + any_present_inflected(text, PERSONNEL_MOVE_STEMS_EN, EN_VERB_SUFFIX)
            + any_present_inflected(text, PERSONNEL_MOVE_STEMS_NORDIC, SV_VERB_SUFFIX)
        )
        if not (role_hits and move_hits):
            return False, None, Details(self.name, {}, 0.0, False)

        name_hit = detect_person_name_spacy(original)
        if not name_hit:
            return False, None, Details(self.name, {}, 0.0, False)

        prox_ok, prox_pair = proximity_cooccurs_morph(
            text,
            PERSONNEL_ROLES,
            PERSONNEL_MOVE_STEMS_EN + PERSONNEL_MOVE_STEMS_NORDIC + PERSONNEL_FIXED_PHRASES,
            window=24
        )
        if not prox_ok:
            return False, None, Details(self.name, {}, 0.0, False)

        toks = tokenize(text)
        name_tok = tokenize(name_hit)[0] if name_hit else ""
        name_idxs = [i for i, t in enumerate(toks) if t == name_tok]

        def _positions_morph(terms, min_stem=3):
            hits = []
            for term in terms:
                ptoks = tokenize(term)
                if not ptoks: continue
                m = len(ptoks)
                for i in range(0, len(toks) - m + 1):
                    ok = True
                    for j, pt in enumerate(ptoks):
                        tt = toks[i + j]
                        if len(pt) >= min_stem:
                            if not tt.startswith(pt): ok = False; break
                        else:
                            if tt != pt: ok = False; break
                    if ok: hits.append(i)
            return hits

        role_anchor_idxs = _positions_morph(PERSONNEL_ROLES)  # keep as-is; phrases & role variants benefit from prefix
        move_anchor_idxs = (
            _positions_inflected_single_token(toks, PERSONNEL_MOVE_STEMS_EN, EN_SUFFIXES)
            + _positions_inflected_single_token(toks, PERSONNEL_MOVE_STEMS_NORDIC, SV_SUFFIXES)
            + _positions_morph(PERSONNEL_FIXED_PHRASES)  # phrases are fine with current matcher
        )
        
        #anchor_idxs = role_anchor_idxs + move_anchor_idxs

        #anchor_idxs = _positions_morph(PERSONNEL_ROLES) + _positions_morph(PERSONNEL_MOVE_STEMS_EN + PERSONNEL_MOVE_STEMS_NORDIC + PERSONNEL_FIXED_PHRASES)
        
        
        # REQUIRE the name to be near BOTH a role and a move anchor
        def _near(a, b, k=30): return any(abs(i - j) <= k for i in a for j in b) if a and b else False
        near_role = _near(name_idxs, role_anchor_idxs, 30)
        near_move = _near(name_idxs, move_anchor_idxs, 30)
        if not (near_role and near_move):
            return False, None, Details(self.name, {}, 0.0, False)

        det = []
        det += list(dict.fromkeys(role_hits))
        det += list(dict.fromkeys(move_hits))
        if prox_ok and prox_pair != ("",""):
            det += list(prox_pair)
        det.append(f"name:{name_hit}")
        det = list(dict.fromkeys(det))

        return True, "personnel_management_change", Details(self.name, {"personnel_management_change": det}, RULE_CONF_BASE[self.name], lock=True)
