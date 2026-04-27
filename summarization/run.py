from summarization.utils import load_json, save_text
from summarization.extractor import select_top_segments
from summarization.llm_summarizer import summarize_with_llm
from question_bank.slide_ingestor import ingest_slides

# Maximum words of slide text to pass to the LLM regardless of slide count.
# Prevents context blowup on decks with many slides.
SLIDE_TEXT_WORD_CAP = 1500


SLIDE_FILE_PATH = "test_slides.pdf"

def main():
    data = load_json("tagging2/data/tagged/tagged_output.json")
    lecture_segments = [
        s for s in data if s["tag"] == "LECTURE_CONTENT"
    ]
    n = len(lecture_segments)
    ratio = 0.30 if n < 150 else 0.40 if n < 300 else 0.50
    selected = select_top_segments(lecture_segments, ratio=ratio)
    print(f"[Summarization] Using ratio {ratio} for {n} segments.")
    summary = summarize_with_llm(selected, slide_text=_build_slide_context(ingest_slides(SLIDE_FILE_PATH)) if SLIDE_FILE_PATH else None)
    save_text("summary.txt", summary)
    print("Summary saved to summary.txt")


def run_summarization_pipeline(tagged_segments, slide_file_path=None):
    print(f"[Summarization] Step 1: Filtering to LECTURE_CONTENT segments...")
    lecture_segments = [
        s for s in tagged_segments if s["tag"] == "LECTURE_CONTENT"
    ]
    print(f"[Summarization] {len(lecture_segments)} lecture segments from {len(tagged_segments)} total.")
    print(f"[Summarization] Step 2: Scoring and extracting top segments...")
    n = len(lecture_segments)
    ratio = 0.60 if n < 150 else 0.70 if n < 300 else 0.80
    target_length = "short" if n < 100 else "long" if n > 250 else "medium"
    print(f"[Summarization] Using ratio {ratio} and target_length '{target_length}' for {n} segments.")
    selected = select_top_segments(lecture_segments, ratio=ratio)
    print(f"[Summarization] {len(selected)} segments selected after scoring and deduplication.")

    slide_text = None
    if slide_file_path:
        try:
            print(f"[Summarization] Step 2b: Ingesting slides from {slide_file_path}...")
            slides = ingest_slides(slide_file_path)
            slide_text = _build_slide_context(slides)
            word_count = len(slide_text.split())
            print(f"[Summarization] {len(slides)} slides ingested, {word_count} words of slide context.")
        except Exception as e:
            print(f"[Summarization] Slide ingestion failed: {e}. Continuing without slides.")
            slide_text = None

    print(f"[Summarization] Step 3: Sending to LLM for rewrite...")
    summary = summarize_with_llm(selected, slide_text=slide_text,target_length=target_length)
    print(f"[Summarization] Done. Summary generated ({len(summary)} chars).")
    return summary


def _build_slide_context(slides: list) -> str:
    """
    Concatenates slide text into a single string up to SLIDE_TEXT_WORD_CAP words.
    Slides are included in order until the cap is hit — early slides take priority
    since lecturers generally front-load key definitions and concepts.
    """
    chunks = []
    total_words = 0

    for slide in slides:
        slide_words = slide["text"].split()
        if total_words + len(slide_words) > SLIDE_TEXT_WORD_CAP:
            # Include as many words from this slide as the cap allows
            remaining = SLIDE_TEXT_WORD_CAP - total_words
            if remaining > 20:
                chunks.append(f"[Slide {slide['slide_number']}] " + " ".join(slide_words[:remaining]))
            break
        chunks.append(f"[Slide {slide['slide_number']}] {slide['text']}")
        total_words += len(slide_words)

    return "\n".join(chunks)


if __name__ == "__main__":
    main()