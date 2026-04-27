# quiz/prompts.py
# Prompt builders for both quiz types.

from typing import Optional

MCQ_BLOOM_LEVELS = ["remember", "understand"]

def build_mcq_prompt(question: str, concept_text: str) -> str:
    """
    Given a QB question and its source concept text, ask the LLM to:
    1. First identify the correct answer (anchored to concept text)
    2. Then generate 3 plausible but wrong distractors
    
    Two-step anchoring reduces nonsense distractors significantly.
    Output must be JSON.
    """

    words = concept_text.split()
    if len(words) > 200:
        concept_text = " ".join(words[:200]) + "..."

    return f"""You are generating a multiple choice question for university students.

Source concept from lecture:
{concept_text}

Question (already generated from this concept):
{question}

Your task:
Step 1 - Write the correct answer to this question based ONLY on the concept above. Keep it concise (one sentence or a short phrase).
Step 2 - Write exactly 3 distractors. Each distractor must:
- Be the same type and category as the correct answer
- Be plausible to a student who partially understands the concept
- Be definitively wrong relative to the concept text
- NOT be "none of the above" or "all of the above"
- NOT repeat or paraphrase the correct answer

Output ONLY this JSON, nothing else:
{{
  "correct": "correct answer here",
  "distractors": ["distractor 1", "distractor 2", "distractor 3"]
}}

JSON Output:"""


def build_extempore_prompt(concept_text: str) -> str:
    """
    Given a concept block from the lecture, generate a single
    presentation-style topic title suitable for a 10-15 min extempore.
    """

    words = concept_text.split()
    if len(words) > 200:
        concept_text = " ".join(words[:200]) + "..."

    return f"""You are generating a presentation topic title for a university student.

The following is a block of lecture content on a specific topic:
{concept_text}

Generate a single concise, academically worded presentation title for this content.
Requirements:
- Must be specific, not generic (avoid titles like "Introduction to X" or "Overview of Y")
- Should be suitable for a 10-15 minute student presentation
- Must reflect the actual depth and focus of the content above
- Output ONLY the title, no explanation, no punctuation at the end

Title:"""