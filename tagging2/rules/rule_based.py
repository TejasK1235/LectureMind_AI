# rules/rule_based.py
from typing import Optional

ADMIN_STRONG_PHRASES = [
    "important for exam",
    "will come in exam",
    "asked in exam",
    "marks distribution",
    "attendance will be taken",
    "assignment submission",
    "internal marks",
    "external marks",
    "question paper",
    "syllabus",
    "exam point of view",
    "from exam perspective"
]

ADMIN_STRONG_KEYWORDS = [
    "exam",
    "attendance",
    "assignment",
    "marks",
    "grading",
    "submission",
    "deadline"
]

CHATTER_PHRASES = [
    "good morning",
    "good afternoon",
    "hello everyone",
    "can you hear me",
    "am i audible",
    "is my screen visible",
    "wait a second",
    "just a moment",
    "any doubts",
    "last class",
    "next class"
]

LECTURE_VERBS = [
    "define",
    "explain",
    "consider",
    "assume",
    "derive",
    "calculate",
    "observe",
    "understand"
]


def apply_rules(text: str) -> Optional[str]:
    t = text.lower().strip()

    # ADMINISTRATIVE (highest priority)
    for phrase in ADMIN_STRONG_PHRASES:
        if phrase in t:
            return "ADMINISTRATIVE"

    for keyword in ADMIN_STRONG_KEYWORDS:
        if keyword in t:
            return "ADMINISTRATIVE"

    # OTHER_CHATTER
    for phrase in CHATTER_PHRASES:
        if phrase in t:
            return "OTHER_CHATTER"


    # LECTURE_CONTENT (fallback only if academic wording exists)
    if any(verb in t for verb in LECTURE_VERBS):
        # avoid meta sentences slipping through
        if "exam" not in t and "important" not in t:
            return "LECTURE_CONTENT"


    return None
    