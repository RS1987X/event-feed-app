# event_taxonomies/rules/personnel.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, any_present, detect_person_name, proximity_cooccurs
from .constants import PERSONNEL_ROLES, PERSONNEL_MOVES, RULE_CONF_BASE

@register
class PersonnelRule:
    name = "role + move + name"
    priority = -30

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        role_hits = any_present(text, PERSONNEL_ROLES)
        move_hits = any_present(text, PERSONNEL_MOVES)
        name_hit = detect_person_name(ctx.text_original)
        if role_hits and move_hits and name_hit:
            prox_ok, prox_pair = proximity_cooccurs(text, PERSONNEL_ROLES, PERSONNEL_MOVES, window=20)
            details = list(set(role_hits + move_hits))
            if prox_ok:
                details += list(prox_pair)
            details.append(f"name:{name_hit}")
            return True, "personnel_management_change", Details(self.name, {"personnel_management_change": details}, RULE_CONF_BASE[self.name], lock=True)
        return False, None, Details(self.name, {}, 0.0, False)
