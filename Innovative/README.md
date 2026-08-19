# 🚀 StartupAdvisorLM — AI Business Idea Validation Assistant

A **Fundamentals of Data Science** project. Type a startup idea in plain English
and the app runs it through a pipeline of classic ML models to return a
validation report and a verdict: **GO / PIVOT / RETHINK**.

It runs **fully offline** on a synthetic (but realistic) dataset — no API keys,
no internet, nothing to pay for.

---

## What it does

You enter an idea + a short plan (team size, planned funding, target market),
and StartupAdvisorLM returns:

- **Category** the idea belongs to (FinTech, EdTech, FoodTech, …)
- **Success odds** for the plan
- **Market sentiment** for that category
- **Ideal customer segment** (who to sell to)
- **Nearest existing ideas** (competitors, and whether they won or failed)
- A blended **0–100 viability score** and a **verdict**
- A **dashboard** of charts + the biggest weakness to fix

---

## The pipeline (and the FODS concept behind each step)

| # | Module | File | Technique |
|---|--------|------|-----------|
| 1 | Text preprocessing | `src/preprocess.py` | cleaning + keyword extraction (`collections.Counter`) |
| 2 | Category classifier | `src/classifier.py` | **TF-IDF + Naive Bayes** (text classification) |
| 3 | Success predictor | `src/predictor.py` | **Logistic Regression** + one-hot encoding |
| 4 | Sentiment engine | `src/sentiment.py` | **Bag-of-words + Naive Bayes** |
| 5 | Customer segmentation | `src/segmentation.py` | **KMeans** + StandardScaler |
| 6 | Competitor search | `src/competitors.py` | **TF-IDF + cosine similarity** |
| 7 | Scoring engine | `src/scoring.py` | weighted multi-signal blend |
| 8 | Dashboard | `src/visuals.py` | **matplotlib** (radar, bars, scatter) |

`src/pipeline.py` wires them together. There are **two front-ends** on the same
brain: the primary **web app** (`api.py` + `web/`, a hand-built Neo-Brutalist UI)
and an optional **Streamlit** UI (`app.py`).

---

## Project structure

```
startup/
├── api.py               # ⭐ FastAPI backend — serves web/ + the ML pipeline
├── web/                 # ⭐ custom Neo-Brutalist frontend
│   ├── index.html
│   ├── style.css        # all the styling / effects
│   └── app.js           # fetches /validate, draws SVG charts
├── app.py               # optional Streamlit UI (alternative front-end)
├── train.py             # trains models, prints held-out accuracy
├── generate_data.py     # builds the synthetic datasets
├── requirements.txt
├── README.md
├── data/                # created by generate_data.py
│   ├── startups.csv
│   ├── reviews.csv
│   └── customers.csv
└── src/                 # the data-science engine (unchanged by the UI)
    ├── preprocess.py
    ├── classifier.py
    ├── predictor.py
    ├── sentiment.py
    ├── segmentation.py
    ├── competitors.py
    ├── scoring.py
    ├── visuals.py
    └── pipeline.py
```

---

## How to run

From the `startup/` folder:

```bash
pip install -r requirements.txt
python generate_data.py           # step 1: create the datasets (run once)
uvicorn api:app --port 8000       # step 2: launch the web app
```

Then open **http://localhost:8000**.

**Windows one-click:** set your key once with `setx GROQ_API_KEY your-key` (then
open a new terminal), and from then on just double-click **`run.bat`** to launch.

Optional extras:

```bash
python train.py                   # print held-out model accuracy
streamlit run app.py              # the alternative Streamlit UI (port 8501)
```

---

## 🧠 Smart Mode (optional LLM layer — free API key)

Tick **Smart Mode** in the UI to add an AI-written advisory section on top of the
numeric report. The classical ML stays the quantitative brain (scores, verdict);
the LLM adds a tailored narrative — strengths, risks, next steps, go-to-market,
and a second opinion on the category (including ideas outside the 8 domains).

- Works with any **OpenAI-compatible** provider, so you can use a **free** key.
- **No SDK to install** — it uses the Python standard library.
- Fully optional and **degrades gracefully**: with no key the toggle disables
  itself and the classical report works unchanged.

### Get a free key (pick one)

| Provider | `LLM_PROVIDER` | Free key from | Default model |
|---|---|---|---|
| **Groq** (fast, no card) | `groq` | https://console.groq.com/keys | `llama-3.3-70b-versatile` |
| **Google Gemini** | `gemini` | https://aistudio.google.com/app/apikey | `gemini-2.0-flash` |
| **OpenRouter** | `openrouter` | https://openrouter.ai/keys | `llama-3.3-70b-instruct:free` |
| **Mistral** | `mistral` | https://console.mistral.ai/api-keys | `mistral-small-latest` |

### Run it (Groq example)

```bash
# bash:
export LLM_PROVIDER="groq"
export LLM_API_KEY="your-free-key"
python -m uvicorn api:app --port 8000
```
PowerShell: `$env:LLM_PROVIDER="groq"` and `$env:LLM_API_KEY="your-free-key"`.

Optional overrides: `LLM_MODEL` (change the model), `LLM_BASE_URL` (custom endpoint).

Backend: `src/advisor.py` + the `/advise` and `/smart_status` endpoints in `api.py`.

---

## Scoring

The final score is a weighted blend of five axes (each mapped to 0–1):

| Axis | Weight | Source |
|------|--------|--------|
| Success | 35% | Logistic Regression on the founder's plan |
| Novelty | 20% | 1 − similarity to the nearest existing idea |
| Sentiment | 20% | Naive Bayes on that category's market reviews |
| Market | 15% | Target market size (1–10) |
| Category | 10% | Historical success rate of the category |

**Verdict:** ≥ 68 → GO · 50–68 → PIVOT · < 50 → RETHINK

---

## Note on the data

`generate_data.py` creates a **synthetic, seeded** dataset (480 labelled ideas,
360 reviews, 350 customers). It's designed so the models have real, learnable
signal — swap in a real Kaggle/Crunchbase-style CSV with the same columns and
everything else works unchanged.
