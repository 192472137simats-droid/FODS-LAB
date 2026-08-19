"""
visuals.py
----------
Module 8: matplotlib figures for the dashboard. Every function returns a
Matplotlib Figure so Streamlit can render it with st.pyplot(fig).

All charts use a transparent background + light text so they blend into the
app's dark theme instead of sitting in white boxes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe inside Streamlit
import matplotlib.pyplot as plt

# ---- palette (matches the app theme) -------------------------------------
ACCENT = "#818cf8"   # indigo
ACCENT2 = "#a78bfa"  # violet
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#334155"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _clean(ax):
    """Transparent background, no top/right spines."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    return ax


def radar_chart(axes_100: dict):
    """Radar/spider chart of the five scoring axes."""
    labels = list(axes_100.keys())
    values = [axes_100[k] for k in labels]
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.4), subplot_kw=dict(polar=True))
    ax.set_facecolor("none")
    ax.plot(angles, values, color=ACCENT, linewidth=2.2)
    ax.fill(angles, values, color=ACCENT, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["", "40", "", "80", ""], fontsize=7, color=MUTED)
    ax.set_ylim(0, 100)
    ax.grid(color=GRID, alpha=0.6)
    ax.spines["polar"].set_color(GRID)
    fig.tight_layout()
    return fig


def category_bar(proba: dict, top_n: int = 5):
    """Horizontal bar chart of the top category probabilities."""
    items = sorted(proba.items(), key=lambda kv: kv[1])[-top_n:]
    names = [k for k, _ in items]
    vals = [v * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _clean(ax)
    colors = [ACCENT if i == len(vals) - 1 else GRID for i in range(len(vals))]
    ax.barh(names, vals, color=colors, height=0.62)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)")
    for i, v in enumerate(vals):
        ax.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=8, color=TEXT)
    fig.tight_layout()
    return fig


def competitor_bar(competitors: list):
    """Horizontal bar chart of similarity to nearest existing ideas."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _clean(ax)
    if not competitors:
        ax.text(0.5, 0.5, "No similar ideas found", ha="center", va="center",
                color=MUTED)
        ax.axis("off")
        return fig

    labels = [f"#{i+1} {c['category']} · {c['outcome']}"
              for i, c in enumerate(competitors)][::-1]
    sims = [c["similarity"] * 100 for c in competitors][::-1]
    colors = [GREEN if c["outcome"] == "Success" else RED for c in competitors][::-1]
    ax.barh(labels, sims, color=colors, height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Similarity (%)")
    for i, v in enumerate(sims):
        ax.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=8, color=TEXT)
    fig.tight_layout()
    return fig


def segment_scatter(seg_bundle, target_idx):
    """Scatter of customers (income vs spend), coloured by segment."""
    df = seg_bundle["data"]
    palette = [ACCENT, ACCENT2, AMBER, GREEN, RED]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    _clean(ax)
    for seg in sorted(df["segment"].unique()):
        part = df[df["segment"] == seg]
        is_target = (seg == target_idx)
        ax.scatter(part["income_k"], part["spend_score"],
                   s=55 if is_target else 16,
                   c=palette[seg % len(palette)],
                   alpha=0.95 if is_target else 0.30,
                   label=f"Seg {seg}" + (" ★" if is_target else ""),
                   edgecolors="white" if is_target else "none",
                   linewidths=0.5)
    ax.set_xlabel("Income (k)")
    ax.set_ylabel("Spend score")
    leg = ax.legend(fontsize=7, loc="best", framealpha=0.15)
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    fig.tight_layout()
    return fig
