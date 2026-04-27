# question_bank/run.py

import json
from typing import List, Dict, Optional

from tagging2.schema import TaggedSegment
from question_bank.concept_builder import build_concepts
from question_bank.concept_scorer import score_concepts
from question_bank.question_generator import generate_questions_for_concept, rewrite_for_variety
from question_bank.evaluator import evaluate_questions
from question_bank.qb_assembler import assemble_question_bank, compute_bloom_counts
from question_bank.slide_ingestor import ingest_slides


def run_qb_pipeline(
    tagged_segments: List[TaggedSegment],
    total_questions: int,
    bloom_percentages: Dict[str, float],
    slide_file_path: Optional[str] = None
) -> Dict:

    slides = None
    if slide_file_path:
        print(f"[QB] Loading slides from: {slide_file_path}")
        try:
            slides = ingest_slides(slide_file_path)
            print(f"[QB] {len(slides)} slides loaded.")
        except Exception as e:
            print(f"[QB] Warning: could not load slides — {e}. Continuing without slides.")
            slides = None

    print("[QB] Step 1: Building concepts...")
    concepts = build_concepts(tagged_segments, slides=slides)    # slides passed here
    print(f"[QB] {len(concepts)} concepts built.")

    # Report how many concepts got slide enrichment
    if slides:
        enriched = sum(1 for c in concepts if c.get("slide_text"))
        print(f"[QB] {enriched}/{len(concepts)} concepts enriched with slide content.")

    print("[QB] Step 2: Scoring concepts...")
    scored_concepts = score_concepts(concepts)

    SCORE_THRESHOLD = 0.15
    filtered_concepts = [c for c in scored_concepts if c["score"] >= SCORE_THRESHOLD]
    if not filtered_concepts:
        half = max(1, len(scored_concepts) // 2)
        filtered_concepts = scored_concepts[:half]
    scored_concepts = filtered_concepts
    print(f"[QB] {len(scored_concepts)} concepts retained after score filtering.")

    print("[QB] Step 3: Computing Bloom distribution...")
    bloom_counts = compute_bloom_counts(total_questions, bloom_percentages)
    print(f"[QB] Distribution: {bloom_counts}")

    print("[QB] Step 4: Generating questions...")
    questions_by_bloom: Dict[str, List[str]] = {level: [] for level in bloom_counts}

    BUFFER_MULTIPLIER = 1.5

    for bloom_level, count_needed in bloom_counts.items():
        if count_needed <= 0:
            continue

        total_needed_with_buffer = int(count_needed * BUFFER_MULTIPLIER) + 1
        total_score = sum(c["score"] for c in scored_concepts) or 1
        accumulated = []

        for concept in scored_concepts:
            share = concept["score"] / total_score
            n = max(1, round(share * total_needed_with_buffer))

            generated = generate_questions_for_concept(
                concept=concept,
                bloom_level=bloom_level,
                num_questions=n
            )
            accumulated.extend(generated)

            # Show whether slide enrichment was used for this concept
            used_slides = "+" if concept.get("slide_text") else " "
            print(f"[QB]   {bloom_level} | concept {concept['concept_id']} [{used_slides}slide] | {len(generated)} questions")

        questions_by_bloom[bloom_level] = accumulated
    
    print("[QB] Step 4b: Rewriting for variety...")
    for bloom_level in questions_by_bloom:
        if questions_by_bloom[bloom_level]:
            original_count = len(questions_by_bloom[bloom_level])
            questions_by_bloom[bloom_level] = rewrite_for_variety(
                questions_by_bloom[bloom_level],
                bloom_level
            )
            print(f"[QB]   {bloom_level}: {original_count} questions rewritten.")

    print("[QB] Step 5: Evaluating and deduplicating...")
    evaluated = evaluate_questions(questions_by_bloom)

    print("[QB] Step 6: Assembling final question bank...")
    result = assemble_question_bank(
        evaluated_questions=evaluated,
        bloom_distribution=bloom_counts,
        scored_concepts=scored_concepts
    )

    if result["warnings"]:
        print("[QB] Warnings:")
        for w in result["warnings"]:
            print(f"  ⚠ {w}")

    result["concepts"] = [
        {
            "concept_id": c["concept_id"],
            "text": c["text"],
            "score": c["score"],
            "word_count": c["word_count"],
            "emphasis_count": c["emphasis_count"]
        }
        for c in scored_concepts
    ]

    return result


# Temporary runner for testing

if __name__ == "__main__":
    import os

    tagged_path = os.path.join(
        os.path.dirname(__file__), "..", "tagging2", "data", "tagged", "tagged_output.json"
    )

    with open(tagged_path, "r", encoding="utf-8") as f:
        tagged_segments = json.load(f)

    TOTAL_QUESTIONS = 35
    BLOOM_PERCENTAGES = {
        "remember":  0.2,
        "understand": 0.2,
        "apply":     0.2,
        "analyze":   0.2,
        "evaluate":  0.2
    }

    SLIDE_FILE_PATH = "test_slides.pdf"

    result = run_qb_pipeline(
        tagged_segments=tagged_segments,
        total_questions=TOTAL_QUESTIONS,
        bloom_percentages=BLOOM_PERCENTAGES,
        slide_file_path=SLIDE_FILE_PATH
    )

    print("\n===== QUESTION BANK OUTPUT =====\n")
    for bloom_level, questions in result["questions"].items():
        print(f"── {bloom_level} ({len(questions)} questions) ──")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        print()

    if result["warnings"]:
        print("Warnings:", result["warnings"])

    output_path = os.path.join(os.path.dirname(__file__), "..", "qb_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Full output saved to qb_output.json")