"""
sentiment.py
------------
Module 4 of the pipeline: market sentiment.

"What does the market already say about this kind of product?"

Technique: a bag-of-words Naive Bayes sentiment classifier trained on labelled
reviews. We then apply it to the reviews of the detected category to estimate
the share of positive sentiment in that market.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .preprocess import clean_text


def _make_pipeline():
    return Pipeline([
        ("bow", CountVectorizer(preprocessor=clean_text, ngram_range=(1, 2))),
        ("nb", MultinomialNB()),
    ])


def build_sentiment_model(reviews_df):
    model = _make_pipeline()
    model.fit(reviews_df["review"], reviews_df["label"])
    return model


def evaluate_sentiment(reviews_df, test_size=0.25, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        reviews_df["review"], reviews_df["label"],
        test_size=test_size, random_state=random_state, stratify=reviews_df["label"])
    model = _make_pipeline().fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


def category_sentiment(model, reviews_df, category):
    """Return (positive_ratio, n_reviews) for a category's market reviews."""
    subset = reviews_df[reviews_df["category"] == category]
    if len(subset) == 0:
        subset = reviews_df
    preds = model.predict(subset["review"])
    pos_ratio = float((preds == "positive").mean())
    return pos_ratio, int(len(subset))
