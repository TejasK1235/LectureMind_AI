# question_bank/question_generator.py

import json
import time
import os
from typing import List, Optional
from groq import Groq
from tagging2.schema import Concept
from tagging2.config import GROQ_MODEL_QB
from question_bank.prompts import build_question_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY4"))
MAX_RETRIES = 2


def generate_questions_for_concept(
    concept: Concept,
    bloom_level: str,
    num_questions: int
) -> List[str]:

    if num_questions <= 0:
        return []

    # Pass slide_text only if the concept has one attached
    slide_text: Optional[str] = concept.get("slide_text", None)

    prompt = build_question_prompt(
        concept_text=concept["text"],
        bloom_level=bloom_level,
        num_questions=num_questions,
        slide_text=slide_text
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_QB,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
            )
            raw = response.choices[0].message.content.strip()
            questions = _parse_response(raw)
            if questions:
                return questions

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"[QB] Rate limit hit, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"[QB] LLM call failed on attempt {attempt + 1}: {e}")

    print(f"[QB] Warning: no questions generated for concept {concept['concept_id']} at {bloom_level}")
    return []


def _parse_response(raw: str) -> List[str]:
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass

    return []

def rewrite_for_variety(questions: List[str], bloom_level: str) -> List[str]:
    """
    Post-generation rewrite pass. Takes a list of questions at a single
    Bloom level and rewrites any that share structural or phrasing patterns
    so the full set reads naturally varied.
    Returns the rewritten list, or the original if the LLM call fails.
    """
    if len(questions) <= 1:
        return questions

    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""You are an experienced university professor reviewing a set of exam questions.

Below are {len(questions)} questions all at the {bloom_level} level of Bloom's Taxonomy.
Your job is to rewrite any questions that share similar sentence structure, opening phrasing, 
or repetitive patterns — so that every question in the final set reads differently.

Rules:
- Preserve the cognitive demand of every question exactly — only change phrasing and structure
- Preserve all subject matter content — do not change what concept is being tested
- Questions that are already naturally phrased and distinct from others should NOT be changed
- Every question must still be at the {bloom_level} Bloom level after rewriting
- Output ONLY a JSON array of strings with exactly {len(questions)} questions in the same order
- No explanation, no numbering, no extra text

Questions to review and rewrite where needed:
{numbered}

JSON Output:"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_QB,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw = response.choices[0].message.content.strip()
            rewritten = _parse_response(raw)
            if rewritten and len(rewritten) == len(questions):
                return rewritten
            else:
                print(f"[QB] Rewrite pass returned wrong count ({len(rewritten)} vs {len(questions)}), keeping original.")
                return questions

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"[QB] Rewrite rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[QB] Rewrite pass failed on attempt {attempt + 1}: {e}")

    print(f"[QB] Rewrite pass failed after all retries, keeping original questions.")
    return questions