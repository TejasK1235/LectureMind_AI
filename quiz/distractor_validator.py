from typing import Optional
from sentence_transformers import SentenceTransformer, util

OPTION_SIMILARITY_THRESHOLD = 0.60

# If a distractor is semantically this similar to the correct answer or its too close a student picking it is essentially picking the right answer
DISTRACTOR_SEMANTIC_THRESHOLD = 0.85

# Reuse the same model already cached from QB evaluator, no extra download needed
_model = SentenceTransformer("all-MiniLM-L6-v2")


def validate_mcq(correct: str, distractors: list) -> Optional[dict]:
    """
    Runs 3 checks on a raw LLM MCQ response:
    1. All 4 options are non-empty distinct strings
    2. No two options are too similar to each other (Jaccard)
    3. Correct answer has some overlap with concept text (groundedness)

    Returns {"correct": str, "distractors": [str, str, str]} if valid, else None.
    """
    if not correct or not correct.strip():
        return None

    if len(distractors) != 3:
        return None

    all_options = [correct.strip()] + [d.strip() for d in distractors]

    if any(not opt for opt in all_options):
        return None

    if len(set(opt.lower() for opt in all_options)) < 4:
        return None

    # Check 1: no two options too similar via Jaccard
    for i in range(len(all_options)):
        for j in range(i + 1, len(all_options)):
            if _jaccard(all_options[i], all_options[j]) >= OPTION_SIMILARITY_THRESHOLD:
                return None

    # Check 2: no distractor semantically too close to correct answer
    correct_embedding = _model.encode(correct.strip(), convert_to_tensor=True)
    for distractor in distractors:
        distractor_embedding = _model.encode(distractor.strip(), convert_to_tensor=True)
        score = util.cos_sim(correct_embedding, distractor_embedding).item()
        if score >= DISTRACTOR_SEMANTIC_THRESHOLD:
            return None

    return {
        "correct": correct.strip(),
        "distractors": [d.strip() for d in distractors]
    }


def _jaccard(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)