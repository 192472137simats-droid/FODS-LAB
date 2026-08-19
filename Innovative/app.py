"""
app.py — StartupAdvisorLM
--------------------------
Streamlit web app: type a business idea, get a data-driven validation report.

Run from the `startup/` folder:
    python generate_data.py      # once, to create the datasets
    streamlit run app.py
"""

import streamlit as st

from src.pipeline import build_engine, validate
from src import visuals

st.set_page_config(page_title="StartupAdvisorLM", page_icon="🚀", layout="wide")

# --------------------------------------------------------------------- style
st.markdown("""
<style>
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden;}
  .block-container {max-width:1080px; padding-top:2.2rem; padding-bottom:3rem;}

  .hero-title {font-size:2.7rem; font-weight:800; line-height:1.05;
      background:linear-gradient(90deg,#818cf8,#c084fc 60%,#f0abfc);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.15rem;}
  .hero-sub {color:#94a3b8; font-size:1.02rem; margin-bottom:.4rem;}

  .sec {font-size:1.15rem; font-weight:700; color:#e2e8f0;
      margin:1.6rem 0 .7rem; padding-left:.6rem; border-left:3px solid #6366f1;}

  .verdict {display:flex; align-items:center; gap:1.6rem; border-radius:18px;
      padding:1.35rem 1.8rem; margin:.3rem 0 1.1rem; color:#fff;
      box-shadow:0 10px 34px rgba(0,0,0,.38);}
  .verdict.green {background:linear-gradient(135deg,#065f46,#10b981);}
  .verdict.amber {background:linear-gradient(135deg,#7c2d12,#f59e0b);}
  .verdict.red   {background:linear-gradient(135deg,#7f1d1d,#ef4444);}
  .v-score {font-size:3.3rem; font-weight:800; line-height:1; white-space:nowrap;}
  .v-score span {font-size:1.05rem; opacity:.72; font-weight:600;}
  .v-label {font-size:1.55rem; font-weight:800; letter-spacing:1.5px;}
  .v-reason {opacity:.92; margin-top:.15rem; font-size:.95rem;}

  .stat {background:#141d2e; border:1px solid #263149; border-radius:14px;
      padding:.95rem 1.05rem; height:100%;}
  .stat-label {color:#8ea0bd; font-size:.72rem; text-transform:uppercase; letter-spacing:.6px;}
  .stat-value {font-size:1.55rem; font-weight:800; color:#e8edf7; margin-top:.15rem; line-height:1.1;}
  .stat-sub {color:#818cf8; font-size:.78rem; margin-top:.25rem;}

  .chip {display:inline-block; background:#1b2740; border:1px solid #334155;
      color:#c7d2fe; padding:.26rem .7rem; border-radius:999px;
      font-size:.8rem; margin:.16rem .16rem 0 0;}

  [data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
  div.stButton>button, div[data-testid="stFormSubmitButton"]>button {
      border-radius:12px; font-weight:700; padding:.55rem 0; font-size:1.02rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the advisor models…")
def get_engine():
    return build_engine()


def stat_card(label, value, sub=""):
    st.markdown(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------- header
st.markdown('<div class="hero-title">🚀 StartupAdvisorLM</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI Business Idea Validation Assistant — a data-science '
            'pipeline that scores your idea and tells you <b>Go · Pivot · Rethink</b>.</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧠 The pipeline")
    st.markdown(
        "- **Category** — TF-IDF + Naive Bayes\n"
        "- **Success** — Logistic Regression\n"
        "- **Sentiment** — bag-of-words model\n"
        "- **Segments** — KMeans clustering\n"
        "- **Competitors** — cosine similarity\n"
        "- **Score** — weighted blend → verdict")
    st.divider()
    st.caption("FODS project · runs fully offline on a synthetic dataset.")

engine = get_engine()

# ----------------------------------------------------------------- input form
st.markdown('<div class="sec">1 · Describe your idea</div>', unsafe_allow_html=True)
with st.container(border=True):
    with st.form("idea_form"):
        idea = st.text_area(
            "Business idea", label_visibility="collapsed",
            placeholder="e.g. An app that delivers home-cooked meals from verified local chefs",
            height=88)
        c1, c2, c3 = st.columns(3)
        team_size = c1.slider("👥 Team size", 1, 25, 8)
        funding_k = c2.slider("💰 Planned funding (₹k)", 5, 600, 150, step=5)
        market_size = c3.slider("📈 Target market size (1–10)", 1, 10, 6)
        submitted = st.form_submit_button("Validate idea 🚀", type="primary",
                                          width="stretch")

# -------------------------------------------------------------------- results
if not submitted:
    st.info("Fill in an idea above and press **Validate idea** to see the report.")
    st.stop()

if len(idea.strip()) < 8:
    st.warning("Please describe your idea in a sentence or two.")
    st.stop()

r = validate(engine, idea, team_size, funding_k, market_size)

# --- verdict card
st.markdown('<div class="sec">2 · Verdict</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="verdict {r["tone"]}">'
    f'<div class="v-score">{r["score"]:.0f}<span>/100</span></div>'
    f'<div class="v-body"><div class="v-label">{r["verdict"]}</div>'
    f'<div class="v-reason">{r["reason"]}</div></div></div>',
    unsafe_allow_html=True)

# --- KPI stat cards
k1, k2, k3, k4 = st.columns(4)
with k1: stat_card("Category", r["category"], f"{r['confidence']:.0%} confident")
with k2: stat_card("Success odds", f"{r['success_prob']:.0%}", "logistic model")
with k3: stat_card("Market sentiment", f"{r['sentiment_pos']:.0%}",
                   f"positive · {r['sentiment_reviews']} reviews")
with k4: stat_card("Novelty", f"{r['axes']['Novelty']:.0f}/100", "less crowded = higher")

if r["keywords"]:
    chips = "".join(f'<span class="chip">{k}</span>' for k in r["keywords"])
    st.markdown(f'<div style="margin-top:.9rem">🔑 {chips}</div>', unsafe_allow_html=True)

# --- dashboard
st.markdown('<div class="sec">3 · Dashboard</div>', unsafe_allow_html=True)
d1, d2 = st.columns(2)
with d1:
    with st.container(border=True):
        st.caption("Validation profile")
        st.pyplot(visuals.radar_chart(r["axes"]), width="stretch")
with d2:
    with st.container(border=True):
        st.caption("Category match")
        st.pyplot(visuals.category_bar(r["proba"]), width="stretch")

d3, d4 = st.columns(2)
with d3:
    with st.container(border=True):
        st.caption("Nearest existing ideas")
        st.pyplot(visuals.competitor_bar(r["competitors"]), width="stretch")
with d4:
    with st.container(border=True):
        st.caption("Customer segments")
        st.pyplot(visuals.segment_scatter(engine["segmenter"], r["segment"]["segment"]),
                  width="stretch")

# --- insights
st.markdown('<div class="sec">4 · Insights</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    with st.container(border=True):
        seg = r["segment"]
        st.markdown("**🎯 Ideal customer segment**")
        st.markdown(f"### {seg['label']}")
        st.caption(f"~{seg['size']} customers · avg age {seg['profile']['age']:.0f} · "
                   f"income ₹{seg['profile']['income_k']:.0f}k · "
                   f"spend {seg['profile']['spend_score']:.0f}/100")
        st.markdown(f"**🛠 Biggest weakness:** `{r['weakest_axis']}` "
                    f"({r['axes'][r['weakest_axis']]:.0f}/100) — focus there.")
with right:
    with st.container(border=True):
        st.markdown("**🥊 Nearest existing ideas**")
        if r["competitors"]:
            st.dataframe(
                [{"Category": c["category"], "Outcome": c["outcome"],
                  "Similarity": f"{c['similarity']*100:.0f}%", "Idea": c["idea"]}
                 for c in r["competitors"]],
                hide_index=True, width="stretch")
        else:
            st.write("No close matches — this idea looks fairly unique.")

with st.expander("How was this scored?  (methodology)"):
    st.markdown(
        "The final score is a weighted blend of five axes:\n\n"
        "| Axis | Weight | Source |\n|---|---|---|\n"
        "| Success | 35% | Logistic Regression on your plan |\n"
        "| Novelty | 20% | 1 − similarity to nearest idea |\n"
        "| Sentiment | 20% | Naive Bayes on market reviews |\n"
        "| Market | 15% | Your target market size |\n"
        "| Category | 10% | Historical success rate of the category |\n")
    st.json(r["axes"])
