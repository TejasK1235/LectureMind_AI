# quiz/extempore_generator.py
# Generates one presentation topic title from a single concept object.
# Returns a topic dict or None if LLM fails.

import os
import time
from typing import Optional

from groq import Groq
from tagging2.config import GROQ_MODEL_EXTEMPORE
from quiz.prompts import build_extempore_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY3"))
MAX_RETRIES = 2


def generate_extempore_topic(concept: dict) -> Optional[dict]:
    """
    Input:
        concept — a single concept object from build_concepts()
                  must have: concept_id, text, score, word_count

    Output:
        {
            "concept_id": "concept_abc123",
            "title": "The Role of Inodes in Linux File System Metadata Management",
            "score": 0.84,
            "word_count": 210
        }
        or None if LLM fails after all retries.
    """

    prompt = build_extempore_prompt(concept["text"])

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_EXTEMPORE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            title = response.choices[0].message.content.strip()

            if not title or len(title.split()) > 25 or len(title.split()) < 3:
                continue

            return {
                "concept_id": concept["concept_id"],
                "title": title,
                "score": concept["score"],
                "word_count": concept["word_count"]
            }

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"[Quiz/Extempore] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[Quiz/Extempore] LLM call failed on attempt {attempt + 1}: {e}")

    print(f"[Quiz/Extempore] Warning: could not generate title for concept {concept['concept_id']}")
    return None