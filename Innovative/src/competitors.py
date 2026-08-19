"""
competitors.py
--------------
Module 6 of the pipeline: competitor / similarity search.

"Which existing ideas in our dataset are closest to this one?"

Technique: TF-IDF vectors + cosine similarity -- a content-based similarity
search over all known ideas. Also yields a 'novelty' signal (how crowded the
space is) that feeds the scoring engine.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text


def build_similarity(startups_df):
    vec = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(startups_df["idea"])
    return {"vectorizer": vec, "matrix": matrix, "df": startups_df.reset_index(drop=True)}


def find_competitors(bundle, text, k=3):
    """Return (list_of_competitors, top_similarity)."""
    q = bundle["vectorizer"].transform([text])
    sims = cosine_similarity(q, bundle["matrix"])[0]
    order = sims.argsort()[::-1]

    results = []
    for i in order:
        # skip a near-identical match to the query itself
        if sims[i] >= 0.999:
            continue
        row = bundle["df"].iloc[i]
        results.append({
            "idea": row["idea"],
            "category": row["category"],
            "outcome": row["outcome"],
            "similarity": round(float(sims[i]), 3),
        })
        if len(results) >= k:
            break

    top_sim = results[0]["similarity"] if results else 0.0
    return results, top_sim
