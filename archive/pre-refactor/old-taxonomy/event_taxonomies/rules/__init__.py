from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Ctx  # only for type checkers; not imported at runtime


@dataclass
class Details:
    rule: str
    hits: Dict[str, List[str]]
    rule_conf: float
    lock: bool
    notes: str = ""
    needs_review_low_sim: bool = False

class Rule(Protocol):
    name: str
    priority: int          # lower runs earlier
    def run(self, ctx: "Ctx", keywords: Dict[str, List[str]]
            ) -> Tuple[bool, Optional[str], Details]: ...

# Global registry of *instances*
REGISTRY = []

def register(cls):
    inst = cls()            # ← instantiate once at import
    REGISTRY.append(inst)   # ← store the instance
    return cls              # keep the class name usable in the module