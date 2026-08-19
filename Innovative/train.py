"""
train.py
--------
Trains every model on a held-out split and prints honest accuracy numbers,
then runs one sample idea through the full pipeline. Great for the report /
viva -- it shows each data-science module actually works.

    python train.py
"""

from src import classifier, predictor, sentiment
from src.pipeline import load_data, build_engine, validate


def main():
    startups, reviews, customers = load_data()

    print("=" * 60)
    print("StartupAdvisorLM — training & evaluation")
    print("=" * 60)
    print(f"startups : {len(startups)} ideas across {startups['category'].nunique()} categories")
    print(f"reviews  : {len(reviews)} market reviews")
    print(f"customers: {len(customers)} customer records")
    print("-" * 60)

    acc_cat = classifier.evaluate_category_classifier(startups)
    acc_succ = predictor.evaluate_success_predictor(startups)
    acc_sent = sentiment.evaluate_sentiment(reviews)

    print("Held-out accuracy")
    print(f"  Category classifier (TF-IDF + Naive Bayes) : {acc_cat:.1%}")
    print(f"  Success predictor   (Logistic Regression)  : {acc_succ:.1%}")
    print(f"  Sentiment model     (BoW + Naive Bayes)    : {acc_sent:.1%}")
    print("-" * 60)

    engine = build_engine()
    demo = "An AI powered app for students to save on food delivery and home cooked meals"
    print(f"Sample idea:\n  \"{demo}\"\n")
    r = validate(engine, demo, team_size=6, funding_k=120, market_size=7)
    print(f"  Category   : {r['category']} ({r['confidence']:.0%} confidence)")
    print(f"  Success P  : {r['success_prob']:.0%}")
    print(f"  Sentiment  : {r['sentiment_pos']:.0%} positive")
    print(f"  Score      : {r['score']}/100  ->  {r['verdict']}")
    print(f"  Target     : {r['segment']['label']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
