# chunk_class.py
from dataclasses import dataclass, asdict

# chunk data structure
@dataclass
class Chunk:
    policy_id: str
    section: str
    subsection: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str

    def to_dict(self):
        return asdict(self)