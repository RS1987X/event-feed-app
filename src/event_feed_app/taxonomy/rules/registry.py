from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

REGISTRY: List[Any] = []

@dataclass
class Details:
    rule: str
    hits: Dict[str, List[str]]
    rule_conf: float = 0.0
    lock: bool = False
    needs_review_low_sim: bool = False
    notes: str = ""

def register(cls):
    REGISTRY.append(cls())
    return cls
