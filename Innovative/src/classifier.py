"""
classifier.py
-------------
Module 2 of the pipeline: the category classifier.

"What kind of business is this idea?"  ->  FinTech / EdTech / FoodTech / ...

Technique: TF-IDF (bag of words + bigrams) feeding a Multinomial Naive Bayes
classifier -- the classic text-classification recipe from FODS.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .preprocess import clean_text


def _make_pipeline():
    return Pipeline([
        # min_df=1 keeps every training word in the vocabulary, so real-world
        # words a user types are far more likely to be recognised.
        # sublinear_tf + a small alpha sharpen the probabilities so a clear
        # idea gets a decisive, confident category instead of a vague guess.
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=1, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


def build_category_classifier(startups_df):
    """Fit on ALL data for deployment inside the app."""
    model = _make_pipeline()
    model.fit(startups_df["idea"], startups_df["category"])
    return model


def evaluate_category_classifier(startups_df, test_size=0.25, random_state=42):
    """Honest held-out accuracy for the training report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        startups_df["idea"], startups_df["category"],
        test_size=test_size, random_state=random_state, stratify=startups_df["category"])
    model = _make_pipeline().fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return acc


def predict_category(model, text):
    """Return (predicted_category, confidence, full_probability_dict)."""
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda kv: -kv[1])
    top_label, top_conf = ranked[0]
    return top_label, float(top_conf), {c: float(p) for c, p in ranked}
