# quiz/run.py
# Entry point for both quiz types.
# 
# run_mcq_quiz()       -> takes QB JSON output -> list of MCQ dicts
# run_extempore_quiz() -> takes concepts list  -> list of topic dicts

import json
import os
from typing import List, Dict, Optional

from quiz.mcq_generator import generate_mcq
from quiz.extempore_generator import generate_extempore_topic
from quiz.prompts import MCQ_BLOOM_LEVELS

# Mirrors the QB pipeline's SCORE_THRESHOLD of 0.15 but slightly higher
EXTEMPORE_SCORE_THRESHOLD = 0.25


def run_mcq_quiz(
    qb_result: Dict,
    concepts: List[Dict],
    num_questions: int = 10
) -> Dict:
    """
    Generates an MCQ quiz from QB output.

    Input:
        qb_result     — the full dict returned by run_qb_pipeline()
        concepts      — scored concept list from the same pipeline run
        num_questions — how many MCQs to attempt to generate

    Output:
        {
            "total":     10,
            "mcqs":      [{question, options, correct, correct_text}, ...],
            "warnings":  ["only 8 valid MCQs generated, requested 10"]
        }
    """

    # Guard: cap num_questions against available candidate pool
    available = sum(
        len(qb_result.get("questions", {}).get(level, []))
        for level in MCQ_BLOOM_LEVELS
    )
    if available == 0:
        return {"total": 0, "mcqs": [], "warnings": [f"No questions found at Bloom levels: {MCQ_BLOOM_LEVELS}"]}
    if num_questions > available:
        print(f"[Quiz/MCQ] Requested {num_questions} but only {available} candidates available. Capping.")
        num_questions = available

    # Step 1: collect candidate questions from allowed Bloom levels only
    candidates = []
    for level in MCQ_BLOOM_LEVELS:
        questions = qb_result.get("questions", {}).get(level, [])
        for q in questions:
            candidates.append({"question": q, "bloom": level})

    if not candidates:
        return {
            "total": 0,
            "mcqs": [],
            "warnings": [f"No questions found at Bloom levels: {MCQ_BLOOM_LEVELS}"]
        }

    # Step 2: build a lookup from concept text
    # We use the first concept's text as a fallback if we can't match
    concept_text_pool = " ".join(c["text"] for c in concepts) if concepts else ""

    # Step 3: generate MCQs up to num_questions
    mcqs = []
    warnings = []

    for candidate in candidates:
        if len(mcqs) >= num_questions:
            break

        result = generate_mcq(
            question=candidate["question"],
            concept_text=concept_text_pool
        )

        if result:
            result["bloom_level"] = candidate["bloom"]
            mcqs.append(result)
            print(f"[Quiz/MCQ] ✓ {candidate['bloom']} | {candidate['question'][:60]}...")
        else:
            print(f"[Quiz/MCQ] ✗ skipped: {candidate['question'][:60]}...")

    if len(mcqs) < num_questions:
        warnings.append(
            f"Only {len(mcqs)} valid MCQs generated, requested {num_questions}"
        )

    return {
        "total": len(mcqs),
        "mcqs": mcqs,
        "warnings": warnings
    }


def run_extempore_quiz(concepts: List[Dict]) -> Dict:
    """
    Generates a pool of presentation topics from scored concepts.

    Input:
        concepts — scored concept list from run_qb_pipeline()
                   (already sorted descending by score)

    Output:
        {
            "total":  8,
            "topics": [
                {"concept_id": "...", "title": "...", "score": 0.84, "word_count": 210},
                ...
            ],
            "warnings": []
        }
    """

    # Filter to only concepts substantial enough for a presentation
    eligible = [c for c in concepts if c["score"] >= EXTEMPORE_SCORE_THRESHOLD and c["word_count"] >= 80]

    if not eligible:
        eligible = concepts[:max(1, len(concepts) // 2)]

    topics = []
    warnings = []

    for concept in eligible:
        result = generate_extempore_topic(concept)
        if result:
            topics.append(result)
            print(f"[Quiz/Extempore] ✓ {result['title']}")
        else:
            print(f"[Quiz/Extempore] ✗ skipped concept {concept['concept_id']}")

    if not topics:
        warnings.append("No extempore topics could be generated.")

    from sentence_transformers import SentenceTransformer, util
    # import torch

    # EXTEMPORE_DEDUP_THRESHOLD = 0.82
    # EXTEMPORE_DEDUP_THRESHOLD = 0.72
    EXTEMPORE_DEDUP_THRESHOLD = 0.65

    if len(topics) > 1:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        titles = [t["title"] for t in topics]
        embeddings = _model.encode(titles, convert_to_tensor=True)
        kept_indices = []
        for i in range(len(topics)):
            is_duplicate = False
            for j in kept_indices:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim >= EXTEMPORE_DEDUP_THRESHOLD:
                    print(f"[Quiz/Extempore] Dedup dropped '{topics[i]['title']}' (similar to '{topics[j]['title']}', sim={sim:.2f})")
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)
        topics = [topics[i] for i in kept_indices]


    # Keep sorted by score descending so frontend can slice top-N easily
    topics.sort(key=lambda t: t["score"], reverse=True)

    return {
        "total": len(topics),
        "topics": topics,
        "warnings": warnings
    }

# Standalone test runner

if __name__ == "__main__":

    NUM_MCQS = 10
    NUM_TOPICS = 5

    qb_path = os.path.join(os.path.dirname(__file__), "..", "qb_output.json")

    with open(qb_path, "r", encoding="utf-8") as f:
        qb_result = json.load(f)

    concepts = qb_result.get("concepts", [])

    if not concepts:
        print("ERROR: No concepts found in QB output. Make sure you ran the updated run_qb_pipeline().")
        exit(1)

    print(f"\n[Quiz] Loaded {len(concepts)} concepts from QB output.")
    print(f"[Quiz] Generating {NUM_MCQS} MCQs and {NUM_TOPICS} extempore topics...\n")

    print("===== MCQ QUIZ =====\n")
    mcq_result = run_mcq_quiz(qb_result, concepts, num_questions=NUM_MCQS)

    for i, mcq in enumerate(mcq_result["mcqs"], 1):
        print(f"Q{i}. [{mcq['bloom_level']}] {mcq['question']}")
        for letter, text in mcq["options"].items():
            marker = " ✓" if letter == mcq["correct"] else ""
            print(f"   {letter}. {text}{marker}")
        print()

    if mcq_result["warnings"]:
        print("Warnings:", mcq_result["warnings"])

    print("\n===== EXTEMPORE TOPICS =====\n")
    extempore_result = run_extempore_quiz(concepts)

    # Slice to NUM_TOPICS, full list is generated, frontend/teacher picks from top N
    top_topics = extempore_result["topics"][:NUM_TOPICS]

    for i, topic in enumerate(top_topics, 1):
        print(f"{i}. {topic['title']}")
        print(f"   score: {topic['score']} | words: {topic['word_count']}")
        print()

    if extempore_result["warnings"]:
        print("Warnings:", extempore_result["warnings"])

    out_path = os.path.join(os.path.dirname(__file__), "..", "quiz_output_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mcq_quiz": {**mcq_result},
            "extempore_quiz": {
                "total": len(top_topics),
                "topics": top_topics,
                "warnings": extempore_result["warnings"]
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"Full output saved to quiz_output_sample.json")
