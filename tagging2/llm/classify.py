from tagging2.llm.llm_client import LLMClient
from tagging2.llm.prompt import build_tagging_prompt

_llm = LLMClient()


def classify_batch(segments):
    """
    segments: list of segment dicts with 'id' and 'text'
    Returns: dict {segment_id: tag}
    """

    prompt = build_tagging_prompt(segments)
    raw_output = _llm.call(prompt)

    results = {}

    for line in raw_output.splitlines():
        if ":" not in line:
            continue

        seg_id, tag = line.split(":", 1)
        results[seg_id.strip()] = tag.strip().upper()

    return results
