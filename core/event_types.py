from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    source: str
    title: str
    timestamp: datetime
    content: str
    metadata: dict