def build_summary_prompt(text: str, slide_text: str = None, target_length: str = "medium") -> str:
    slide_section = ""
    if slide_text:
        slide_section = f"""
Lecture slide notes (use to fill gaps the transcript may have missed,
such as formulas, definitions, or structured content not fully spoken aloud):
{slide_text}
"""
    length_instruction = {
        "short":  "Aim for 500-600 words total.",
        "medium": "Aim for 700-900 words total.",
        "long":   "Aim for 1000-1400 words total. Include all key points, definitions, and concepts discussed. This will be used as exam revision notes so completeness matters more than brevity."
    }.get(target_length, "Aim for 700-900 words total.")


    return f"""
You are an academic assistant creating detailed revision notes for university students.
Task:
Convert the following lecture excerpts into comprehensive, structured revision notes.
A student should be able to read ONLY these notes and understand everything covered in the lecture.
Rules:
- Cover EVERY distinct topic, concept, definition, and technique mentioned in the excerpts
- Do NOT summarise or compress — if something was discussed, it must appear in the notes
- Do NOT add new information beyond what is in the transcript or slides below
- Use clear headings for each major topic
- Under each heading use bullet points for individual facts, steps, or definitions
- Include formulas, algorithms, and specific values exactly as mentioned
- Keep the tone precise and suitable for exam revision
- If slide notes are provided, use them to add definitions or formulas not spoken aloud
- {length_instruction}
Limit motivational or importance-related content to at most one short section.
{slide_section}Lecture excerpts:
{text}
""".strip()


def build_chunk_prompt(text: str) -> str:
    return f"""
You are an academic assistant.
Task:
Extract all key points, definitions, concepts, and facts from the following lecture excerpts.
Rules:
- Output ONLY a bullet point list, no headings, no prose
- Each bullet point should express exactly one distinct idea
- Do NOT repeat the same idea twice even if the lecturer mentioned it multiple times
- Include every distinct fact, definition, formula, and concept mentioned
- Keep each bullet point concise but complete
- Do not add information not present in the excerpts
Lecture excerpts:
{text}
""".strip()


def build_merge_prompt(chunks_text: str, target_length: str = "medium") -> str:
    length_instruction = {
        "short":  "Aim for 500-600 words total.",
        "medium": "Aim for 700-900 words total.",
        "long":   "Aim for 1000-1400 words total."
    }.get(target_length, "Aim for 700-900 words total.")

    return f"""
You are an academic assistant creating detailed revision notes for university students.
Task:
Below are raw bullet points extracted from different parts of a lecture. Many points will be duplicate or near-duplicate because the lecturer repeated concepts. Your job is to produce clean, well-written revision notes from this raw material.
Rules:
- Identify the major topics covered and create one heading per topic
- Under each heading write concise, clear bullet points covering what was taught
- AGGRESSIVELY merge and eliminate duplicates — if the same concept appears multiple times in different wording, write it ONCE in the best possible wording
- Do NOT keep two bullets that express the same idea even if worded differently
- Write in a clear academic tone — not robotic lists of definitions, but precise and readable
- Include formulas, algorithms, and specific values exactly as mentioned
- Do NOT add information not present in the bullet points below
- {length_instruction}
Limit motivational or importance-related content to at most one short section.
Raw extracted points:
{chunks_text}
""".strip()

