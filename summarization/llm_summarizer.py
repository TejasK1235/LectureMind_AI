import time
from groq import Groq
from summarization.prompt import build_summary_prompt, build_chunk_prompt, build_merge_prompt
from tagging2.config import GROQ_MODEL_SUMMARIZATION
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY3"))

# Max segments per chunk: keeps each LLM call safely under 5000 tokens
CHUNK_SIZE = 70


def _call_llm(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_SUMMARIZATION,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 15 * (attempt + 1)
                print(f"[Summarization] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM call failed after 3 retries")

def _semantic_dedup_bullets(bullets: list, threshold: float = 0.82) -> list:
    if len(bullets) <= 1:
        return bullets
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(bullets, convert_to_tensor=True)
    kept_indices = []
    for i in range(len(bullets)):
        is_duplicate = False
        for j in kept_indices:
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(i)
    return [bullets[i] for i in kept_indices]

def summarize_with_llm(segments, slide_text=None, target_length="medium"):
    if len(segments) <= CHUNK_SIZE:
        # Short enough — single call as before
        combined_text = "\n".join(s["text"] for s in segments)
        prompt = build_summary_prompt(combined_text, slide_text=slide_text, target_length=target_length)
        print(f"[Summarization] Single LLM call ({len(segments)} segments).")
        return _call_llm(prompt)

    # Chunked path, split into CHUNK_SIZE batches, extract bullets, then merge
    chunks = [segments[i:i + CHUNK_SIZE] for i in range(0, len(segments), CHUNK_SIZE)]
    print(f"[Summarization] Chunked into {len(chunks)} batches of ~{CHUNK_SIZE} segments each.")

    bullet_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_text = "\n".join(s["text"] for s in chunk)
        prompt = build_chunk_prompt(chunk_text)
        print(f"[Summarization] Processing chunk {idx + 1}/{len(chunks)}...")
        bullets = _call_llm(prompt)
        bullet_chunks.append(bullets)
        if idx < len(chunks) - 1:
            time.sleep(3)

    # Semantic dedup across all bullets before merge
    print(f"[Summarization] Deduplicating bullets across chunks...")
    all_bullets = []
    for chunk_bullets in bullet_chunks:
        for line in chunk_bullets.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                all_bullets.append(line.lstrip("-*• ").strip())

    deduped_bullets = _semantic_dedup_bullets(all_bullets)
    print(f"[Summarization] {len(all_bullets)} bullets → {len(deduped_bullets)} after dedup.")

    combined_bullets = "\n".join(f"- {b}" for b in deduped_bullets)

    # Final merge call
    print(f"[Summarization] Merging into final notes...")

    # Append slide text to merge prompt if present
    if slide_text:
        combined_bullets += f"\n\n[Slide Notes]\n{slide_text}"

    merge_prompt = build_merge_prompt(combined_bullets, target_length=target_length)
    result = _call_llm(merge_prompt)
    return result