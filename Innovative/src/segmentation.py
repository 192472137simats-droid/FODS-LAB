"""
segmentation.py
---------------
Module 5 of the pipeline: target-customer segmentation.

"Who is the ideal customer for this idea?"

Technique: KMeans clustering (exactly the FODS customer-segmentation setup) on
age / income / spend / online-activity. Each cluster is auto-described in plain
English, and we pick the cluster that best matches the idea's category.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "income_k", "spend_score", "online_activity"]

# Rough "ideal buyer" profile per category: (age, income_k, spend, online)
CATEGORY_PROFILE = {
    "FinTech":    (32, 80, 55, 70),
    "EdTech":     (22, 35, 60, 80),
    "FoodTech":   (30, 55, 70, 75),
    "HealthTech": (40, 70, 50, 55),
    "Ecommerce":  (30, 55, 75, 80),
    "SaaS":       (38, 95, 45, 60),
    "Gaming":     (23, 30, 70, 90),
    "Travel":     (33, 75, 65, 65),
}


def _describe(profile):
    """Turn a centroid into a short human label."""
    age, income, spend, online = profile
    age_word = "young" if age < 28 else ("older" if age > 45 else "mid-age")
    inc_word = "low-income" if income < 45 else ("high-income" if income > 75 else "mid-income")
    spend_word = "frugal" if spend < 45 else ("big spenders" if spend > 65 else "moderate spenders")
    online_word = "very online" if online > 65 else ("mostly offline" if online < 40 else "semi-online")
    return f"{age_word.capitalize()}, {inc_word}, {online_word} {spend_word}"


def build_segmenter(customers_df, k=4, random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(customers_df[FEATURES])
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    df = customers_df.copy()
    df["segment"] = labels
    centroids = df.groupby("segment")[FEATURES].mean()
    descriptions = {int(i): _describe(tuple(centroids.loc[i])) for i in centroids.index}
    sizes = df["segment"].value_counts().to_dict()

    return {
        "scaler": scaler, "kmeans": km, "data": df,
        "centroids": centroids, "descriptions": descriptions,
        "sizes": {int(k_): int(v) for k_, v in sizes.items()},
    }


def target_segment(bundle, category):
    """Nearest cluster (in scaled space) to the category's ideal buyer profile."""
    profile = CATEGORY_PROFILE.get(category)
    if profile is None:
        profile = tuple(bundle["data"][FEATURES].mean())
    p_df = pd.DataFrame([list(profile)], columns=FEATURES)
    p_scaled = bundle["scaler"].transform(p_df)[0]
    centers = bundle["kmeans"].cluster_centers_
    dists = np.linalg.norm(centers - p_scaled, axis=1)
    idx = int(np.argmin(dists))
    return {
        "segment": idx,
        "label": bundle["descriptions"][idx],
        "size": bundle["sizes"][idx],
        "profile": bundle["centroids"].loc[idx].round(1).to_dict(),
    }
