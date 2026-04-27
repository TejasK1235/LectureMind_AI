def build_tagging_prompt(segments):
    instructions = """
You are classifying classroom lecture transcript segments.
Allowed tags (choose exactly one per segment):
- LECTURE_CONTENT
- ADMINISTRATIVE
- OTHER_CHATTER

Definitions:

LECTURE_CONTENT:
Sentences that teach subject matter knowledge. This includes
explanations, definitions, formulas, derivations, worked examples,
conceptual reasoning, and rhetorical questions used to introduce
or explain a concept (e.g. "What does this mean? It means that...").
Even if a sentence starts with "note that", "for example", or
"what is", tag it LECTURE_CONTENT if it contains technical subject matter.

ADMINISTRATIVE:
Sentences about logistics, not about subject knowledge.
ONLY tag ADMINISTRATIVE if the sentence explicitly mentions:
exams, assignments, deadlines, attendance, marks, grading,
syllabus coverage, office hours, or class schedule.
Do NOT tag as ADMINISTRATIVE just because a sentence says something
is "important" or tells students to "remember" a concept —
those are still LECTURE_CONTENT.

OTHER_CHATTER:
Greetings, jokes, technical interruptions (mic issues, screen sharing),
or purely social speech with zero subject matter content.
Do NOT tag as OTHER_CHATTER if the segment contains any technical
subject matter, even if it also contains filler words or questions.

Rules:
- One tag per segment
- When in doubt between LECTURE_CONTENT and another tag, choose LECTURE_CONTENT
- Output format MUST be:
  segment_id: TAG
- One line per segment
- No explanations
"""
    body = "\n".join(
        f"{seg['id']}: {seg['text']}" for seg in segments
    )
    return instructions.strip() + "\n\n" + body