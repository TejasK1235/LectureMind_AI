# pipeline/run.py
import json
import os
from tagging2.pipeline.segment import segment_transcript
from tagging2.pipeline.tagger import tag_segments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_pipeline(input_path: str, output_path: str):

    input_path = os.path.join(BASE_DIR, input_path)
    output_path = os.path.join(BASE_DIR, output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    segments = segment_transcript(raw_segments)
    tagged_segments = tag_segments(segments)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tagged_segments, f, indent=2, ensure_ascii=False)


def run_tagging_pipeline(transcripts):
    """
    Input: list of raw transcript segments (dicts)
    Output: tagged segments (list of dicts)
    """
    from tagging2.pipeline.segment import segment_transcript
    from tagging2.pipeline.tagger import tag_segments

    segments = segment_transcript(transcripts)
    tagged_segments = tag_segments(segments)

    output_path = os.path.join(
        BASE_DIR, "data", "tagged", "tagged_output.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tagged_segments, f, indent=2, ensure_ascii=False)

    return tagged_segments



if __name__ == "__main__":
    run_pipeline(
        input_path="data/raw/transcripts.json",
        output_path="data/tagged/tagged_output.json"
    )
