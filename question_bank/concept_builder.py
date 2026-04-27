# question_bank/concept_builder.py

import uuid
from typing import List, Optional, Dict
from tagging2.schema import TaggedSegment, Concept

CONCEPT_WORD_LIMIT = 250

# Minimum Jaccard overlap for a slide to be considered relevant to a concept.
SLIDE_MATCH_THRESHOLD = 0.08

# Words to ignore when computing overlap too common to be meaningful signals
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "and", "or", "for", "with", "on", "at", "by", "as", "that", "this",
    "it", "its", "be", "has", "have", "we", "you", "they", "i", "so",
    "but", "from", "not", "can", "will", "also", "which", "what", "how"
}


def build_concepts(
    tagged_segments: List[TaggedSegment],
    slides: Optional[List[Dict]] = None
) -> List[Concept]:
    """
    Groups tagged segments into concept blocks.
    If slides are provided, each concept is enriched with the most
    relevant slide's text using word overlap alignment.

    slides parameter is optional — if not provided, works exactly as before.
    """

    concepts = []
    current_bucket: List[TaggedSegment] = []
    last_lecture_end_time: float = None

    def flush(bucket):
        if not bucket:
            return
        text = " ".join(s["text"] for s in bucket)
        concepts.append({
            "concept_id": f"concept_{uuid.uuid4().hex[:8]}",
            "segments": list(bucket),
            "text": text,
            "word_count": len(text.split()),
            "emphasis_count": sum(1 for s in bucket if s.get("emphasis", False)),
            "score": 0.0,
            "slide_text": None
        })

    for seg in tagged_segments:
        if seg["tag"] != "LECTURE_CONTENT":
            if seg["tag"] == "ADMINISTRATIVE":
                flush(current_bucket)
                current_bucket = []
                last_lecture_end_time = None
            continue

        if last_lecture_end_time is not None:
            gap = seg["start_time"] - last_lecture_end_time
            if gap >= 30.0:
                flush(current_bucket)
                current_bucket = []

        current_bucket.append(seg)
        last_lecture_end_time = seg["end_time"]

        current_word_count = sum(len(s["text"].split()) for s in current_bucket)
        if current_word_count >= CONCEPT_WORD_LIMIT:
            flush(current_bucket)
            current_bucket = []
            last_lecture_end_time = None

    flush(current_bucket)

    # Slide alignment (only runs if slides were provided)
    if slides:
        _align_slides_to_concepts(concepts, slides)

    return concepts


def _align_slides_to_concepts(concepts: List[Dict], slides: List[Dict]):
    """
    For each concept, finds the slide with highest word overlap
    and attaches its text to concept["slide_text"].

    Modifies concepts in place.
    """

    # Pre-tokenize all slides once
    slide_word_sets = []
    for slide in slides:
        words = set(slide["text"].lower().split()) - _STOPWORDS
        slide_word_sets.append(words)

    for concept in concepts:
        concept_words = set(concept["text"].lower().split()) - _STOPWORDS

        if not concept_words:
            continue

        best_score = 0.0
        best_slide_text = None

        for i, slide_words in enumerate(slide_word_sets):
            if not slide_words:
                continue

            intersection = concept_words & slide_words
            union = concept_words | slide_words
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard > best_score:
                best_score = jaccard
                best_slide_text = slides[i]["text"]

        # Only attach if overlap is meaningful enough
        if best_score >= SLIDE_MATCH_THRESHOLD:
            concept["slide_text"] = best_slide_text