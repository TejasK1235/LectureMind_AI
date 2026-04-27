import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from summarization.utils import clean_text


def compute_scores(segments):
    texts = [clean_text(s["text"]) for s in segments]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    tfidf_scores = tfidf_matrix.mean(axis=1).A1

    seen_terms = set()
    first_mention_scores = []

    vocab = vectorizer.get_feature_names_out()

    for row in tfidf_matrix:
        indices = row.nonzero()[1]
        terms = {vocab[i] for i in indices}
        new_terms = terms - seen_terms
        seen_terms.update(terms)

        first_mention_scores.append(
            len(new_terms) / max(len(terms), 1)
        )

    n = len(segments)
    position_scores = [1 - (i / n) for i in range(n)]

    length_scores = []
    for s in segments:
        length = len(s["text"].split())
        if 10 <= length <= 40:
            length_scores.append(1.0)
        else:
            length_scores.append(0.5)

    final_scores = (
        0.50 * tfidf_scores +
        0.25 * np.array(first_mention_scores) +
        0.15 * np.array(position_scores) +
        0.10 * np.array(length_scores)
    )

    return final_scores
