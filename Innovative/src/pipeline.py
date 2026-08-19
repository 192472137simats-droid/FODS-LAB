"""
pipeline.py
-----------
The orchestrator. Loads data, builds every model once, and exposes a single
`validate()` call that runs an idea end-to-end through all modules and returns
one tidy report dictionary for the UI.
"""

import os
import pandas as pd

from . import classifier, predictor, sentiment, segmentation, competitors, scoring
from .preprocess import extract_keywords

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


def load_data():
    startups = pd.read_csv(os.path.join(DATA_DIR, "startups.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    return startups, reviews, customers


def build_engine():
    """Fit all models and bundle them with the data they need at inference."""
    startups, reviews, customers = load_data()

    category_base = (
        startups.assign(win=(startups["outcome"] == "Success").astype(int))
        .groupby("category")["win"].mean().to_dict()
    )

    return {
        "startups": startups,
        "reviews": reviews,
        "customers": customers,
        "category_base": category_base,
        "classifier": classifier.build_category_classifier(startups),
        "predictor": predictor.build_success_predictor(startups),
        "sentiment": sentiment.build_sentiment_model(reviews),
        "segmenter": segmentation.build_segmenter(customers),
        "similarity": competitors.build_similarity(startups),
    }


def validate(engine, idea_text, team_size, funding_k, market_size):
    """Run one idea through the whole pipeline -> report dict."""
    idea_text = (idea_text or "").strip()

    # 1. keywords
    keywords = extract_keywords(idea_text)

    # 2. category
    category, confidence, proba = classifier.predict_category(
        engine["classifier"], idea_text)

    # 3. success probability
    success_prob = predictor.predict_success(
        engine["predictor"], category, team_size, funding_k, market_size)

    # 4. market sentiment
    sent_pos, n_reviews = sentiment.category_sentiment(
        engine["sentiment"], engine["reviews"], category)

    # 5. competitors + novelty
    comps, top_sim = competitors.find_competitors(engine["similarity"], idea_text, k=3)

    # 6. target customer segment
    segment = segmentation.target_segment(engine["segmenter"], category)

    # 7. score + verdict
    cat_base = float(engine["category_base"].get(category, 0.5))
    score = scoring.compute_score(success_prob, sent_pos, market_size, cat_base, top_sim)

    return {
        "idea": idea_text,
        "keywords": keywords,
        "category": category,
        "confidence": confidence,
        "proba": proba,
        "success_prob": success_prob,
        "sentiment_pos": sent_pos,
        "sentiment_reviews": n_reviews,
        "competitors": comps,
        "top_similarity": top_sim,
        "segment": segment,
        "category_base": cat_base,
        **score,  # score, verdict, tone, reason, axes, weakest_axis
    }
