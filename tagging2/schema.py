# schema.py
from typing import TypedDict, List, Optional


class TranscriptSegment(TypedDict):
    id: str
    start_time: float
    end_time: float
    text: str


class TaggedSegment(TranscriptSegment):
    tag: str
    emphasis: bool


class Concept(TypedDict):
    concept_id: str
    segments: List[TaggedSegment]
    text: str           # all segment texts joined with space
    word_count: int
    emphasis_count: int
    score: float        # filled in by concept scorer, 0 by default
    slide_text: Optional[str]