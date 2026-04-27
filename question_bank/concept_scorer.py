# question_bank/concept_scorer.py

from typing import List
from tagging2.schema import Concept


# Weights for scoring
WEIGHT_WORD_COUNT = 0.6
WEIGHT_EMPHASIS = 0.4

# Soft cap: concepts above this word count don't get extra credit
WORD_COUNT_CAP = 250


def score_concepts(concepts: List[Concept]) -> List[Concept]:
    if not concepts:
        return []

    word_counts = [min(c["word_count"], WORD_COUNT_CAP) for c in concepts]
    emphasis_counts = [c["emphasis_count"] for c in concepts]

    max_wc = max(word_counts) or 1
    max_em = max(emphasis_counts) or 1

    for i, concept in enumerate(concepts):
        norm_wc = word_counts[i] / max_wc
        norm_em = emphasis_counts[i] / max_em

        # recency_index: 0.0 for first concept, 1.0 for last concept
        recency_index = i / max(len(concepts) - 1, 1)
        recency_bonus = 0.1 * recency_index

        concept["score"] = round(
            WEIGHT_WORD_COUNT * norm_wc + WEIGHT_EMPHASIS * norm_em + recency_bonus, 4
        )

    concepts.sort(key=lambda c: c["score"], reverse=True)

    return concepts