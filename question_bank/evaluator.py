# question_bank/evaluator.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer, util


JACCARD_THRESHOLD = 0.65
SEMANTIC_THRESHOLD = 0.85

_model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_questions(
    questions_by_bloom: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Input:  {"Remember": ["q1", "q2", ...], "Analyze": [...], ...}
    Output: same structure but with duplicates removed and invalid questions filtered.

    Deduplication is done GLOBALLY across all Bloom levels —
    so if the same question appears under Remember and Analyze, one gets dropped.
    """

    # Step 1: flatten all questions with their bloom level tag
    all_questions: List[Dict] = []
    for bloom, questions in questions_by_bloom.items():
        for q in questions:
            all_questions.append({"bloom": bloom, "text": q})

    # Step 2: filter invalid questions
    all_questions = [q for q in all_questions if _is_valid(q["text"])]

    # Step 3: deduplicate globally
    deduplicated = _deduplicate(all_questions)

    # Step 4: rebuild by bloom level
    result: Dict[str, List[str]] = {bloom: [] for bloom in questions_by_bloom}
    for q in deduplicated:
        bloom = q["bloom"]
        if bloom in result:
            result[bloom].append(q["text"])

    return result


def _is_valid(question: str) -> bool:
    """
    Basic sanity checks:
    - Must be long enough to be a real question (at least 6 words)
    - Must end with a question mark OR start with a question word / instruction verb
    """
    words = question.strip().split()
    
    # reject questions that are suspiciously short in terms of content words
    if len([w for w in words if len(w) > 3]) < 4:
        return False

    question_starters = (
        "what", "why", "how", "when", "where", "which", "who",
        "define", "explain", "describe", "compare", "analyze",
        "evaluate", "discuss", "differentiate", "illustrate"
    )

    text_lower = question.strip().lower()
    starts_with_question_word = any(text_lower.startswith(w) for w in question_starters)
    ends_with_question_mark = question.strip().endswith("?")

    return starts_with_question_word or ends_with_question_mark


def _deduplicate(questions: List[Dict]) -> List[Dict]:
    if not questions:
        return []

    # Encode all question texts in one batch
    texts = [q["text"] for q in questions]
    embeddings = _model.encode(texts, convert_to_tensor=True)

    selected = []
    selected_embeddings = []

    for i, candidate in enumerate(questions):
        candidate_words = set(candidate["text"].lower().split())

        too_similar = False

        for j, kept in enumerate(selected):
            # Fast Jaccard check first — if it passes, then do the more expensive semantic check. Avoids unnecessary embedding comparisons.
            kept_words = set(kept["text"].lower().split())
            intersection = candidate_words & kept_words
            union = candidate_words | kept_words
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard >= JACCARD_THRESHOLD:
                too_similar = True
                break

            # Semantic check
            cosine_score = util.cos_sim(
                embeddings[i], selected_embeddings[j]
            ).item()

            if cosine_score >= SEMANTIC_THRESHOLD:
                too_similar = True
                break

        if not too_similar:
            selected.append(candidate)
            selected_embeddings.append(embeddings[i])

    return selected