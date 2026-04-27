# question_bank/qb_assembler.py

from typing import List, Dict, Tuple
from tagging2.schema import Concept


def assemble_question_bank(
    evaluated_questions: Dict[str, List[str]],
    bloom_distribution: Dict[str, int],
    scored_concepts: List[Concept]
) -> Dict:
    """
    Assembles the final question bank.

    Args:
        evaluated_questions: {"Remember": ["q1", ...], "Analyze": [...]}
        bloom_distribution:  {"Remember": 3, "Analyze": 4, ...}  ← exact counts per level
        scored_concepts:     concept list sorted by score descending (for metadata)

    Returns:
        {
            "total_questions": 10,
            "bloom_distribution": {"Remember": 3, "Analyze": 4, ...},
            "questions": {
                "Remember": ["q1", "q2", "q3"],
                "Analyze":  ["q4", "q5", "q6", "q7"]
            },
            "warnings": [...]   ← if requested count couldn't be met for some level
        }
    """

    final_questions: Dict[str, List[str]] = {}
    warnings: List[str] = []
    total = 0

    for bloom_level, requested_count in bloom_distribution.items():
        if requested_count <= 0:
            final_questions[bloom_level] = []
            continue

        available = evaluated_questions.get(bloom_level, [])
        selected = available[:requested_count]
        final_questions[bloom_level] = selected
        total += len(selected)

        if len(selected) < requested_count:
            shortfall = requested_count - len(selected)
            warnings.append(
                f"{bloom_level}: requested {requested_count}, "
                f"only {len(selected)} available (short by {shortfall})"
            )

    ALL_BLOOM_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate"]
    full_distribution = {level: len(final_questions.get(level, [])) for level in ALL_BLOOM_LEVELS}
    full_questions = {level: final_questions.get(level, []) for level in ALL_BLOOM_LEVELS}

    return {
        "total_questions": total,
        "bloom_distribution": full_distribution,
        "questions": full_questions,
        "warnings": warnings
    }

def compute_bloom_counts(
    total_questions: int,
    bloom_percentages: Dict[str, float]
) -> Dict[str, int]:
    """
    Converts percentage-based Bloom distribution into exact question counts.
    Handles rounding so total always equals requested total.

    Args:
        total_questions: e.g. 10
        bloom_percentages: e.g. {"Remember": 0.3, "Analyze": 0.4, "Evaluate": 0.3}
                           Values should sum to 1.0

    Returns:
        {"Remember": 3, "Analyze": 4, "Evaluate": 3}
    """

    counts: Dict[str, float] = {
        level: pct * total_questions
        for level, pct in bloom_percentages.items()
        if pct > 0
    }

    # Floor everything first
    floored: Dict[str, int] = {k: int(v) for k, v in counts.items()}
    remainders: List[Tuple[str, float]] = sorted(
        [(k, counts[k] - floored[k]) for k in counts],
        key=lambda x: x[1],
        reverse=True
    )

    # Distribute leftover slots to highest remainders
    assigned = sum(floored.values())
    leftover = total_questions - assigned

    for i in range(leftover):
        level = remainders[i % len(remainders)][0]
        floored[level] += 1

    return floored