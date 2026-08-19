"""
scoring.py
----------
Module 7 of the pipeline: the scoring engine.

Blends every signal into one 0-100 viability score and a verdict.
Five axes, each mapped to 0-1 then combined with fixed weights:

    Success   -> P(success) from the predictor
    Sentiment -> share of positive market reviews
    Market    -> founder's target market size (1-10 -> 0-1)
    Category  -> historical success base-rate of the category
    Novelty   -> 1 - similarity to nearest existing idea (less crowded = better)
"""

WEIGHTS = {
    "Success": 0.35,
    "Sentiment": 0.20,
    "Market": 0.15,
    "Category": 0.10,
    "Novelty": 0.20,
}


def compute_score(success_prob, sentiment_pos, market_size, category_base, top_similarity):
    axes = {
        "Success": float(success_prob),
        "Sentiment": float(sentiment_pos),
        "Market": float(market_size) / 10.0,
        "Category": float(category_base),
        "Novelty": 1.0 - float(top_similarity),
    }
    score = sum(WEIGHTS[k] * axes[k] for k in WEIGHTS) * 100.0
    score = round(score, 1)

    if score >= 68:
        verdict, tone = "GO", "green"
        reason = "Strong signals across success odds, market and demand."
    elif score >= 50:
        verdict, tone = "PIVOT", "amber"
        reason = "Promising, but one or two axes are dragging it down."
    else:
        verdict, tone = "RETHINK", "red"
        reason = "Weak fundamentals — rework the idea or the plan."

    # weakest axis, for an actionable tip
    weakest = min(axes, key=lambda k: axes[k])
    axes_100 = {k: round(v * 100, 1) for k, v in axes.items()}
    return {
        "score": score,
        "verdict": verdict,
        "tone": tone,
        "reason": reason,
        "axes": axes_100,
        "weakest_axis": weakest,
    }
