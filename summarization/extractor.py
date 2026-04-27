import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarization.scorer import compute_scores
from summarization.utils import clean_text


def select_top_segments(segments, ratio=0.25, similarity_threshold=0.85):
    """
    Select top segments by importance while suppressing semantic redundancy.
    """

    scores = compute_scores(segments)

    texts = [clean_text(s["text"]) for s in segments]

    # TF-IDF vectors for similarity comparison
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    scored = list(enumerate(zip(segments, scores)))
    scored.sort(key=lambda x: x[1][1], reverse=True)

    k = math.ceil(len(scored) * ratio)

    selected_indices = []

    for idx, (segment, score) in scored:
        if len(selected_indices) >= k:
            break

        if not selected_indices:
            selected_indices.append(idx)
            continue

        similarities = cosine_similarity(
            tfidf_matrix[idx],
            tfidf_matrix[selected_indices]
        )[0]

        if max(similarities) < similarity_threshold:
            selected_indices.append(idx)

    selected_segments = [segments[i] for i in selected_indices]
    selected_segments.sort(key=lambda s: s["start_time"])

    return selected_segments
