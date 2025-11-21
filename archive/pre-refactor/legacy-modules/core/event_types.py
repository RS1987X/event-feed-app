from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    source: str
    title: str
    timestamp: datetime
    fetched_at: datetime
    content: str
    metadata: dict

