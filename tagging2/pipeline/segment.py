# pipeline/segment.py
import uuid
from typing import List
from tagging2.schema import TranscriptSegment


def segment_transcript(raw_segments: List[dict]) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []

    for seg in raw_segments:
        text = seg["text"].strip()
        if not text:
            continue

        segments.append({
            "id": f"seg_{uuid.uuid4().hex[:8]}",
            "start_time": float(seg["start"]),
            "end_time": float(seg["end"]),
            "text": text
        })

    return segments
