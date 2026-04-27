from typing import List

from tagging2.schema import TranscriptSegment, TaggedSegment
from tagging2.rules.rule_based import apply_rules
from tagging2.llm.classify import classify_batch


BATCH_SIZE = 10

# how many following lecture segments receive emphasis
EMPHASIS_WINDOW = 3

def tag_segments(segments: List[TranscriptSegment]) -> List[TaggedSegment]:
    """
    Main tagging pipeline:
    1. Rule-based tagging
    2. LLM resolution of undecided segments (batched)
    3. Post-processing: emphasis propagation
    """

    # Step 1: rule-based tagging
    print(f"[Tagging] Step 1: Applying rule-based classifier to {len(segments)} segments...")
    tagged: List[TaggedSegment] = []

    for seg in segments:
        rule_tag = apply_rules(seg["text"])
        tagged.append({
            **seg,
            "tag": rule_tag if rule_tag else "UNDECIDED",
            "emphasis": False
        })

    decided = sum(1 for t in tagged if t["tag"] != "UNDECIDED")
    undecided = len(tagged) - decided
    print(f"[Tagging] Rule-based: {decided} decided, {undecided} sent to LLM.")

    # Step 2: resolve UNDECIDED using batches
    print(f"[Tagging] Step 2: Resolving {undecided} undecided segments via LLM (batches of {BATCH_SIZE})...")
    buffer = []

    for seg in tagged:
        if seg["tag"] == "UNDECIDED":
            buffer.append(seg)
        else:
            _resolve_buffer(buffer)
            buffer.clear()

    _resolve_buffer(buffer)

    from collections import Counter
    dist = Counter(t["tag"] for t in tagged)
    print(f"[Tagging] Tag distribution: {dict(dist)}")

    # Step 3: emphasis linkage
    print(f"[Tagging] Step 3: Running emphasis propagation...")
    _propagate_emphasis(tagged)

    emphasis_count = sum(1 for t in tagged if t.get("emphasis"))
    print(f"[Tagging] Emphasis flagged on {emphasis_count} segments ({round(emphasis_count/len(tagged)*100, 1)}% of total).")
    print(f"[Tagging] Done. {len(tagged)} segments tagged.")

    return tagged

def _resolve_buffer(buffer):
    """Resolve undecided segments using LLM classification."""
    if not buffer:
        return

    for i in range(0, len(buffer), BATCH_SIZE):
        batch = buffer[i:i + BATCH_SIZE]
        results = classify_batch(batch)

        for seg in batch:
            # fallback: treat uncertainty as ADMINISTRATIVE
            seg["tag"] = results.get(seg["id"], "ADMINISTRATIVE")


def _propagate_emphasis(tagged: List[TaggedSegment]):
    """
    If an ADMINISTRATIVE segment appears,
    mark nearby LECTURE_CONTENT segments as emphasized.
    """

    n = len(tagged)

    for i, seg in enumerate(tagged):

        if seg["tag"] != "ADMINISTRATIVE":
            continue

        # propagate emphasis forward
        steps = 0
        j = i + 1

        while j < n and steps < EMPHASIS_WINDOW:
            if tagged[j]["tag"] == "LECTURE_CONTENT":
                tagged[j]["emphasis"] = True
                steps += 1
            elif tagged[j]["tag"] == "OTHER_CHATTER":
                # ignore chatter, don't count window
                pass
            else:
                # stop if another admin appears
                break

            j += 1
