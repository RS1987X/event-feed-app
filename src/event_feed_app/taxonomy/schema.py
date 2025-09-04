from dataclasses import dataclass
from typing import List
# ---- canonical data shapes ----
@dataclass(frozen=True)
class Category:
    cid: str
    name: str
    desc: str

@dataclass(frozen=True)
class Taxonomy:
    version: str
    lang: str
    categories: List[Category]

    @property
    def ordered_cids(self) -> List[str]:
        return [c.cid for c in self.categories]

    @property
    def cid2name(self) -> dict:
        return {c.cid: c.name for c in self.categories}
