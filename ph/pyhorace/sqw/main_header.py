from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class MainHeader:
    """
    Main header information for an SQW file, tracking provenance.
    """
    filename: str = ""
    filepath: str = ""
    title: str = ""
    nfiles: int = 0
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
