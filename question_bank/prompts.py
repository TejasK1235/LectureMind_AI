# question_bank/prompts.py

from typing import Optional

BLOOM_DESCRIPTIONS = {
    "remember":   "recall facts, definitions, and basic concepts directly stated in the text",
    "understand": "explain ideas or concepts in your own words, interpret meaning",
    "apply":      "use the concept in a new or practical situation",
    "analyze":    "break down the concept, examine relationships, identify causes or structure",
    "evaluate":   "make a judgment, critique, or justify a position based on the concept",
    "create":     "propose, design, or formulate something new based on the concept"
}

BLOOM_EXAMPLES = {
    "remember": [
        "What is the primary function of an inode in a Unix filesystem?",
        "Name the two memory regions where Java strings can be stored.",
        "Which data structure is used to implement a priority queue in Dijkstra's algorithm?",
        "State the condition under which a process is considered deadlocked.",
    ],
    "understand": [
        "Why does Java treat String as a class rather than a primitive data type?",
        "What is the significance of storing string literals in the string pool rather than the heap?",
        "How does the concept of virtual memory allow a process to use more memory than is physically available?",
        "In the context of graph traversal, what distinguishes depth-first search from breadth-first search?",
    ],
    "apply": [
        "A program frequently creates the string literal 'admin' across multiple methods. What memory behaviour would you expect, and how would this affect object equality checks?",
        "Given a file system where inodes have a fixed number of direct block pointers, how would you store a file that exceeds the capacity of those direct pointers?",
        "You are designing a cache for a web server that serves repeated requests for the same resource. Which memory management strategy from this lecture would you apply, and why?",
        "If you needed to find the shortest path in a graph with negative edge weights, which algorithm would you choose over Dijkstra's and what modification would it require?",
    ],
    "analyze": [
        "What accounts for the difference in behaviour when comparing two String objects created with new versus two string literals with the same value?",
        "How does the placement of data in heap memory versus the string pool affect garbage collection behaviour in Java?",
        "Why does Bellman-Ford require exactly V-1 iterations to guarantee correctness on a graph with V vertices?",
        "What structural property of a balanced BST makes it more suitable than an unsorted array for implementing a priority queue?",
    ],
    "evaluate": [
        "Under what conditions would using new String() be preferable to a string literal in Java, despite the additional memory cost?",
        "Is the inode-based file system design an effective approach for handling very large files? What are its limitations?",
        "A team proposes using Dijkstra's algorithm on a road network where some roads have tolls represented as negative weights. Assess the validity of this approach.",
        "Compare the tradeoffs between using a contiguous block allocation strategy versus linked allocation for storing files on disk.",
    ],
    "create": [
        "Design a memory management strategy for a system that frequently creates and discards short-lived string objects, minimising garbage collection overhead.",
        "Propose a modification to the standard inode structure that would allow more efficient storage of very small files.",
        "How would you redesign the string pool mechanism to support strings that are frequently modified without causing excessive object creation?",
        "Develop an approach for detecting negative weight cycles in a graph before running a shortest-path algorithm.",
    ],
}

BLOOM_STARTERS = {
    "remember": [
        "What is...",
        "Define...",
        "List...",
        "State...",
        "Name...",
        "Which of the following describes...",
        "What does ... refer to?",
        "Identify...",
        "When does ... occur?",
        "What are the steps involved in...?",
    ],
    "understand": [
        "Explain why...",
        "Describe how...",
        "What does ... mean?",
        "Summarize...",
        "In your own words, what is...?",
        "How would you interpret...?",
        "What is the relationship between ... and ...?",
        "Why is ... important in this context?",
        "How does ... work?",
        "What is the significance of...?",
    ],
    "apply": [
        "How would you use ... in a situation where...?",
        "Given that ..., how would you determine...?",
        "Illustrate how ... applies when...",
        "If you were to implement ..., what steps would you follow?",
        "How could ... be used to solve...?",
        "Demonstrate how ... would behave when...",
        "What would happen if ... were applied to...?",
        "Using the concept of ..., how would you approach...?",
        "How would you modify ... to achieve...?",
        "In what situation would ... be most useful?",
    ],
    "analyze": [
        "What are the differences between ... and ...?",
        "Why does ... cause ...?",
        "Break down the relationship between...",
        "What are the implications of...?",
        "How does ... affect ...?",
        "What factors contribute to...?",
        "Compare and contrast ... and ...",
        "Why is ... structured the way it is?",
        "What would change if ... were removed?",
        "How does ... depend on ...?",
    ],
    "evaluate": [
        "To what extent is ... true?",
        "Assess the validity of...",
        "Critique the claim that...",
        "Is ... an appropriate approach? Why?",
        "Justify why...",
        "What are the strengths and weaknesses of...?",
        "Would you consider ... an effective solution? Why or why not?",
        "How would you judge the effectiveness of...?",
        "Is ... always the best choice? Under what conditions might it fail?",
        "What evidence supports or contradicts the idea that...?",
    ],
    "create": [
        "Design a ... that...",
        "Propose a method to...",
        "Formulate a hypothesis about...",
        "How would you construct...?",
        "What approach would you take to build...?",
        "Suggest an improvement to ... that addresses...",
        "How would you redesign ... to better handle...?",
        "Develop a strategy for...",
        "What would a new system for ... look like?",
        "Propose an alternative to ... that solves...",
    ],
}

MAX_CONCEPT_WORDS = 200
MAX_SLIDE_WORDS = 150    # slide text is usually denser so slightly shorter cap


def build_question_prompt(
    concept_text: str,
    bloom_level: str,
    num_questions: int,
    slide_text: Optional[str] = None
) -> str:

    bloom_desc = BLOOM_DESCRIPTIONS.get(bloom_level, "think critically about the concept")
    starters = BLOOM_STARTERS.get(bloom_level, [])
    starters_text = ", ".join(f'"{s}"' for s in starters)

    # Trim transcript text
    words = concept_text.split()
    if len(words) > MAX_CONCEPT_WORDS:
        concept_text = " ".join(words[:MAX_CONCEPT_WORDS]) + "..."

    if slide_text:
        # Trim slide text
        slide_words = slide_text.split()
        if len(slide_words) > MAX_SLIDE_WORDS:
            slide_text = " ".join(slide_words[:MAX_SLIDE_WORDS]) + "..."

        content_section = f"""Lecture Transcript (what the instructor said):
{concept_text}

Slide Content (what was shown on screen during this section):
{slide_text}

Generate questions that draw from both sources above where relevant."""

    else:
        content_section = f"""Lecture Content:
{concept_text}"""

    examples = BLOOM_EXAMPLES.get(bloom_level, [])
    examples_text = "\n".join(f'- "{q}"' for q in examples)

    return f"""You are an experienced university professor writing exam questions for a final assessment paper. Your questions are clear, natural, and varied — they read like questions a human educator would write, not like AI output.

Your task is to generate exactly {num_questions} exam question(s) based ONLY on the content provided below.

Bloom's Taxonomy Level: {bloom_level}
What this means: Questions must require students to {bloom_desc}.

Here are examples of well-written questions at this Bloom level. Match this style and quality — notice how they vary in structure and read naturally:
{examples_text}

Cognitive anchors for this level (use these to calibrate your thinking, NOT as required opening phrases): {starters_text}

Rules:
- Use ONLY information present in the provided content. Do not add outside knowledge.
- Each question must genuinely be at the {bloom_level} level — not easier, not harder.
- Questions must be descriptive or short-answer style. No MCQs.
- Each question must be a complete, clearly worded sentence.
- Every question must have a different structure and opening — no two questions should start with the same word or phrase.
- Output ONLY a JSON array of strings. No explanation, no numbering, no extra text.

Example output format:
["Question one here?", "Question two here?"]

{content_section}

JSON Output:"""