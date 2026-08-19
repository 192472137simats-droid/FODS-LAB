"""
generate_data.py
-----------------
Creates the synthetic datasets that power StartupAdvisorLM.

We don't have a real startup dataset, so we build three realistic, *labelled*
CSV files that the machine-learning modules can learn from:

    data/startups.csv   -> idea text, category, features, Success/Fail outcome
    data/reviews.csv    -> market reviews per category, positive/negative label
    data/customers.csv  -> customer records used for KMeans segmentation

Everything is seeded, so running this again reproduces the exact same data.
Run once before training:  python generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Vocabulary per business category (signature keywords make ideas learnable)
# ---------------------------------------------------------------------------
# Rich, real-world vocabulary per category. Each list mixes the literal
# category name, common single-word terms, and multi-word phrases so the
# classifier recognises how people ACTUALLY phrase ideas (e.g. "saas tools",
# "remote teams", "fintech") instead of only the narrow synthetic phrases.
CATEGORIES = {
    "FinTech":    ["payments", "payment app", "digital wallet", "banking", "loans", "micro loans",
                   "credit score", "investing", "budgeting", "insurance", "upi", "transactions",
                   "fintech", "finance", "money", "expense tracking", "mutual funds", "stocks",
                   "savings", "lending", "financial", "neobank", "credit card", "tax filing"],
    "EdTech":     ["online courses", "tutoring", "live tutoring", "exam prep", "skill learning", "classroom",
                   "quizzes", "certification", "study groups", "coding lessons", "language learning",
                   "edtech", "education", "e-learning", "upskilling", "mentorship", "courses",
                   "learning app", "test prep", "homework help", "virtual classroom", "students"],
    "FoodTech":   ["home cooked meals", "food delivery", "recipes", "cloud kitchen", "local chefs",
                   "grocery delivery", "meal planning", "restaurant booking", "healthy snacks", "tiffin service",
                   "foodtech", "food", "dining", "cooking", "meals", "groceries", "catering",
                   "meal kits", "food ordering", "street food"],
    "HealthTech": ["fitness coaching", "teleconsultation", "mental wellness", "diet planning", "doctor appointments",
                   "medicine delivery", "therapy sessions", "health tracking", "yoga classes", "patient records",
                   "healthtech", "healthcare", "telemedicine", "mental health", "workout", "nutrition",
                   "symptom checker", "wellness", "fitness", "medical"],
    "Ecommerce":  ["online store", "marketplace", "handmade products", "fashion retail", "electronics deals",
                   "subscription boxes", "local sellers", "dropshipping", "product discovery", "brand outlet",
                   "ecommerce", "shopping", "retail", "products", "deals", "d2c", "online shopping",
                   "reselling", "boutique", "storefront"],
    "SaaS":       ["analytics dashboard", "workflow automation", "team productivity", "crm", "crm software",
                   "invoicing tool", "project management", "customer support software", "data reporting",
                   "scheduling tool", "hr platform", "saas", "software", "b2b", "tools", "business tools",
                   "automation", "dashboard", "collaboration", "remote teams", "productivity software",
                   "workspace", "enterprise software", "api integration", "team management"],
    "Gaming":     ["mobile game", "multiplayer arena", "esports tournaments", "puzzle game", "play to earn rewards",
                   "game streaming", "fantasy sports", "casual games", "game community", "level challenges",
                   "gaming", "game", "players", "tournaments", "gamers", "arcade", "game studio",
                   "metaverse", "leaderboards", "in game rewards"],
    "Travel":     ["trip planning", "hotel booking", "flight deals", "travel itinerary", "local tours",
                   "homestays", "backpacking", "vacation packages", "road trips", "tourism guide",
                   "travel", "trips", "hotels", "flights", "booking", "destinations", "holidays",
                   "getaways", "travel booking", "tour packages"],
}

AUDIENCE = ["students", "small businesses", "urban families", "young professionals", "freelancers",
            "seniors", "gamers", "travelers", "home cooks", "startups", "women", "teenagers"]
VERBS = ["discover", "manage", "save on", "connect with", "learn", "book",
         "track", "automate", "share", "compare", "access", "organize"]
ADJ = ["AI powered", "affordable", "on demand", "hyperlocal", "next gen",
       "peer to peer", "personalized", "community driven", "smart", "all in one"]
PLATFORM = ["app", "platform", "service", "website", "startup", "product", "solution"]

# Historical base success rate per category (drives the outcome label)
BASE_RATE = {
    "FinTech": 0.58, "EdTech": 0.50, "FoodTech": 0.44, "HealthTech": 0.56,
    "Ecommerce": 0.50, "SaaS": 0.62, "Gaming": 0.40, "Travel": 0.45,
}


def make_idea(cat, cross_noise=0.28):
    """Compose a natural-sounding idea sentence for a category.

    With probability `cross_noise` we swap in ONE keyword borrowed from a
    different category. This creates genuinely ambiguous ideas so the
    classifier lands at a realistic (~90-95%) accuracy instead of 100%.
    """
    kws = random.sample(CATEGORIES[cat], k=random.choice([2, 3]))
    if random.random() < cross_noise:
        other = random.choice([c for c in CATEGORIES if c != cat])
        kws[random.randrange(len(kws))] = random.choice(CATEGORIES[other])
    a = kws[0]
    b = kws[1] if len(kws) > 1 else kws[0]
    template = random.choice([
        "A {adj} {plat} for {aud} to {verb} {a} and {b}.",
        "An {adj} {plat} that helps {aud} with {a} and {b}.",
        "A {plat} offering {a}, {b} and more for {aud}.",
        "A {adj} {plat} that lets {aud} {verb} {a} through {b}.",
    ])
    return template.format(adj=random.choice(ADJ), plat=random.choice(PLATFORM),
                           aud=random.choice(AUDIENCE), verb=random.choice(VERBS), a=a, b=b)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 2. startups.csv  (idea + features + outcome)
# ---------------------------------------------------------------------------
def build_startups(n_per_cat=60):
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_cat):
            team_size = int(np.clip(np.random.normal(9, 4), 1, 25))
            funding_k = float(np.clip(np.random.lognormal(mean=4.9, sigma=0.6), 5, 600))
            market_size = int(np.clip(round(np.random.normal(6, 2)), 1, 10))

            # Outcome from a logistic model of the features + category base rate
            z = (-0.4
                 + 3.0 * (BASE_RATE[cat] - 0.5)
                 + 1.2 * ((team_size - 10) / 10.0)
                 + 1.4 * ((funding_k - 150) / 150.0)
                 + 1.7 * ((market_size - 5) / 5.0)
                 + np.random.normal(0, 0.8))
            outcome = "Success" if np.random.rand() < sigmoid(z) else "Fail"

            rows.append({
                "idea": make_idea(cat),
                "category": cat,
                "team_size": team_size,
                "funding_k": round(funding_k, 1),
                "market_size": market_size,
                "outcome": outcome,
            })
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. reviews.csv  (market sentiment per category)
# ---------------------------------------------------------------------------
POS_PHRASES = ["is amazing", "works great", "saves me so much time", "is very reliable",
               "is easy to use", "is worth every rupee", "is super helpful", "is a game changer",
               "runs smoothly", "has excellent support"]
NEG_PHRASES = ["is terrible", "keeps crashing", "is way too slow", "is confusing to use",
               "is a waste of money", "has poor support", "is overpriced", "is full of bugs",
               "is disappointing", "is frustrating"]


def build_reviews(n_per_cat=45, mixed_noise=0.16):
    """Build labelled market reviews.

    A fraction `mixed_noise` of reviews are 'mixed' -- they contain both a
    positive and a negative phrase, so their label is genuinely ambiguous.
    This keeps the sentiment model realistic (~88-94%) instead of perfect.
    """
    rows = []
    for cat, kws in CATEGORIES.items():
        # Category positivity loosely tracks its success base rate
        pos_ratio = np.clip(BASE_RATE[cat] + np.random.normal(0, 0.05), 0.30, 0.80)
        for _ in range(n_per_cat):
            kw = random.choice(kws)
            if np.random.rand() < mixed_noise:
                # ambiguous review: praise + complaint in one sentence
                text = (f"The {kw} {random.choice(POS_PHRASES)}, "
                        f"but it {random.choice(NEG_PHRASES)}.")
                label = random.choice(["positive", "negative"])
            elif np.random.rand() < pos_ratio:
                text = f"The {kw} feature {random.choice(POS_PHRASES)}."
                label = "positive"
            else:
                text = f"The {kw} {random.choice(NEG_PHRASES)}."
                label = "negative"
            rows.append({"category": cat, "review": text, "label": label})
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. customers.csv  (for KMeans customer segmentation)
# ---------------------------------------------------------------------------
def build_customers(n=350):
    """Four latent personas blended into one customer table."""
    personas = [
        # (age, income_k, spend_score, online_activity)
        (24, 30, 75, 85),   # young, low income, high spend, very online
        (38, 90, 55, 55),   # mid-career, high income, moderate
        (55, 60, 35, 25),   # older, offline, cautious spender
        (30, 55, 65, 70),   # urban professional, balanced-high
    ]
    rows = []
    for _ in range(n):
        base = personas[np.random.randint(len(personas))]
        age = int(np.clip(np.random.normal(base[0], 6), 18, 70))
        income = int(np.clip(np.random.normal(base[1], 15), 12, 160))
        spend = int(np.clip(np.random.normal(base[2], 12), 1, 100))
        online = int(np.clip(np.random.normal(base[3], 12), 1, 100))
        rows.append({"age": age, "income_k": income,
                     "spend_score": spend, "online_activity": online})
    return pd.DataFrame(rows)


def main():
    startups = build_startups()
    reviews = build_reviews()
    customers = build_customers()

    startups.to_csv(os.path.join(DATA_DIR, "startups.csv"), index=False)
    reviews.to_csv(os.path.join(DATA_DIR, "reviews.csv"), index=False)
    customers.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)

    print("Generated datasets in", DATA_DIR)
    print(f"  startups.csv  : {len(startups):4d} rows  "
          f"({startups['outcome'].value_counts().to_dict()})")
    print(f"  reviews.csv   : {len(reviews):4d} rows  "
          f"({reviews['label'].value_counts().to_dict()})")
    print(f"  customers.csv : {len(customers):4d} rows")


if __name__ == "__main__":
    main()
