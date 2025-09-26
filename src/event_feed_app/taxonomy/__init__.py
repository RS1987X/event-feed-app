from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple, Any

# --- simple registry ---
_RULES: List[type] = []

def register(cls):
    _RULES.append(cls)
    return cls

@dataclass
class Details:
    rule: str
    hits: Dict[str, List[str]]
    rule_conf: float
    lock: bool
    needs_review_low_sim: bool = False

def get_rules():
    # instantiate once in declared order (priority is inside each rule)
    return [RuleClass() for RuleClass in _RULES]
