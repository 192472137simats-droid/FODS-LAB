"""
predictor.py
------------
Module 3 of the pipeline: the success predictor.

"Given the category and the founder's plan (team size, funding, market size),
how likely is this to succeed?"

Technique: Logistic Regression on numeric features + one-hot encoded category.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

NUMERIC = ["team_size", "funding_k", "market_size"]


def _design_matrix(df):
    """Numeric features + one-hot category -> feature DataFrame."""
    dummies = pd.get_dummies(df["category"], prefix="cat")
    X = pd.concat([df[NUMERIC].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    return X


def build_success_predictor(startups_df):
    X = _design_matrix(startups_df)
    y = (startups_df["outcome"] == "Success").astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return {"model": model, "columns": list(X.columns)}


def evaluate_success_predictor(startups_df, test_size=0.25, random_state=42):
    X = _design_matrix(startups_df)
    y = (startups_df["outcome"] == "Success").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


def predict_success(bundle, category, team_size, funding_k, market_size):
    """Return P(success) in [0, 1] for a hypothetical startup."""
    row = {c: 0 for c in bundle["columns"]}
    row["team_size"] = team_size
    row["funding_k"] = funding_k
    row["market_size"] = market_size
    cat_col = f"cat_{category}"
    if cat_col in row:
        row[cat_col] = 1
    X = pd.DataFrame([row])[bundle["columns"]]
    return float(bundle["model"].predict_proba(X)[0][1])
