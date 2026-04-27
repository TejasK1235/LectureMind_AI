# quiz/mcq_generator.py
# Generates one validated MCQ from a single QB question + concept text.
# Returns a structured MCQ dict or None if validation fails after retries.

import json
import os
import random
import time
from typing import Optional

from groq import Groq
from tagging2.config import GROQ_MODEL_MCQ
from quiz.prompts import build_mcq_prompt
from quiz.distractor_validator import validate_mcq

client = Groq(api_key=os.getenv("GROQ_API_KEY3"))
MAX_RETRIES = 2


def generate_mcq(question: str, concept_text: str) -> Optional[dict]:
    """
    Input:
        question     — a single question string from the QB output
        concept_text — the concept text the question was generated from

    Output:
        {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "correct": "B",         <- letter of correct answer
            "correct_text": "..."   <- actual text of correct answer
        }
        or None if LLM fails or validation fails after all retries.
    """

    prompt = build_mcq_prompt(question, concept_text)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_MCQ,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            # print(f"[DEBUG] Raw LLM response:\n{raw}\n")
            parsed = _parse_response(raw)

            if not parsed:
                continue

            validated = validate_mcq(
                correct=parsed["correct"],
                distractors=parsed["distractors"],
            )

            if not validated:
                continue

            return _build_mcq_dict(question, validated)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"[Quiz/MCQ] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[Quiz/MCQ] LLM call failed on attempt {attempt + 1}: {e}")

    print(f"[Quiz/MCQ] Warning: could not generate valid MCQ for: {question[:60]}...")
    return None


def _parse_response(raw: str) -> Optional[dict]:
    """Strip markdown fences if present, parse JSON."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if (
            isinstance(parsed, dict)
            and "correct" in parsed
            and "distractors" in parsed
            and isinstance(parsed["distractors"], list)
        ):
            return parsed
    except json.JSONDecodeError:
        pass

    return None


def _build_mcq_dict(question: str, validated: dict) -> dict:
    """
    Shuffles correct answer among distractors, assigns A/B/C/D labels,
    records which letter is correct.
    """
    all_options = [validated["correct"]] + validated["distractors"]
    random.shuffle(all_options)

    labels = ["A", "B", "C", "D"]
    options = {labels[i]: all_options[i] for i in range(4)}

    correct_letter = next(
        label for label, text in options.items()
        if text == validated["correct"]
    )

    return {
        "question": question,
        "options": options,
        "correct": correct_letter,
        "correct_text": validated["correct"]
    }