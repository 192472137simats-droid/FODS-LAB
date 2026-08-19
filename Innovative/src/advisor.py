"""
advisor.py — LLM-augmented "Smart Mode" (free providers)
--------------------------------------------------------
Optional layer that puts a real LLM on top of the classical ML pipeline. The ML
modules stay the quantitative brain (scores, verdict); the LLM adds the
qualitative, tailored narrative — and can give a second opinion on the category,
including ideas that fall outside the 8 supported domains.

It talks to any **OpenAI-compatible** chat endpoint, so you can use a FREE key
from Groq, Google Gemini, OpenRouter, Mistral, etc. Uses only the Python
standard library — nothing to pip install.

Configure with environment variables (all optional):
    LLM_PROVIDER   groq | gemini | openrouter | mistral | openai   (default: groq)
    LLM_API_KEY    your key (or the provider-specific var below)
    LLM_MODEL      override the model name
    LLM_BASE_URL   override the endpoint (for a custom / self-hosted provider)

Provider-specific key vars also work: GROQ_API_KEY, GEMINI_API_KEY,
OPENROUTER_API_KEY, MISTRAL_API_KEY, OPENAI_API_KEY.
"""

import os
import json
import urllib.request
import urllib.error

DEFAULT_PROVIDER = "groq"

# base_url is an OpenAI-compatible root; we POST to <base_url>/chat/completions
PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key_envs": ["GROQ_API_KEY"],
        "signup": "https://console.groq.com/keys",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "signup": "https://aistudio.google.com/app/apikey",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_envs": ["OPENROUTER_API_KEY"],
        "signup": "https://openrouter.ai/keys",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "key_envs": ["MISTRAL_API_KEY"],
        "signup": "https://console.mistral.ai/api-keys",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_envs": ["OPENAI_API_KEY"],
        "signup": "https://platform.openai.com/api-keys",
    },
}

SYSTEM_PROMPT = (
    "You are StartupAdvisorLM's senior startup advisor. You receive a founder's "
    "business idea plus the quantitative output of a data-science scoring pipeline "
    "(detected category, success probability, market sentiment, novelty, nearest "
    "competitors, target customer segment, and a 0-100 viability score with a "
    "GO / PIVOT / RETHINK verdict). Write concise, specific, actionable advice that "
    "is GROUNDED in those numbers — reference the actual signals (e.g. if sentiment "
    "is low, address it; if novelty is low, the space is crowded; if success odds "
    "are low, tie it to the team/funding/market inputs). Never invent metrics. If "
    "the idea seems to fit a different category than the one detected, or falls "
    "outside the eight supported domains (FinTech, EdTech, FoodTech, HealthTech, "
    "Ecommerce, SaaS, Gaming, Travel), say so plainly in category_note.\n\n"
    "Respond with ONLY a JSON object — no prose, no code fences — with exactly "
    "these keys:\n"
    '  one_liner: a crisp one-sentence repositioning of the idea\n'
    '  verdict_take: 1-2 sentences reacting to the score and verdict\n'
    '  strengths: array of ~3 short strings\n'
    '  risks: array of ~3 short strings\n'
    '  suggestions: array of ~3 short, actionable next steps\n'
    '  ideal_customer: one sentence, richer than the clustering label\n'
    '  gtm: array of ~2-3 go-to-market ideas\n'
    '  category_note: one sentence; note if the true category differs or is out of scope\n'
    "Keep every string to one or two sentences."
)


def _resolve():
    """Resolve provider / endpoint / model / key from the environment."""
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    preset = PRESETS.get(provider, {})
    base_url = os.getenv("LLM_BASE_URL") or preset.get("base_url")
    model = os.getenv("LLM_MODEL") or preset.get("model")
    key = os.getenv("LLM_API_KEY")
    if not key:
        for env in preset.get("key_envs", []):
            if os.getenv(env):
                key = os.getenv(env)
                break
    return {"provider": provider, "base_url": base_url, "model": model,
            "key": key, "signup": preset.get("signup")}


def smart_available():
    """(bool, reason) — is Smart Mode usable right now?"""
    cfg = _resolve()
    if not cfg["base_url"]:
        return False, (f"Unknown provider '{cfg['provider']}'. Set LLM_PROVIDER to one of: "
                       + ", ".join(PRESETS) + " — or set LLM_BASE_URL for a custom endpoint.")
    if not cfg["key"]:
        hint = "LLM_API_KEY"
        if cfg["provider"] in PRESETS:
            hint += f" (or {PRESETS[cfg['provider']]['key_envs'][0]})"
        msg = f"No API key set for provider '{cfg['provider']}'. Set {hint}."
        if cfg.get("signup"):
            msg += f" Get a free key at {cfg['signup']}"
        return False, msg
    return True, "ready"


def _build_user_prompt(r):
    comps = "; ".join(
        f"{c['category']} ({c['outcome']}, {round(c['similarity']*100)}% similar)"
        for c in r.get("competitors", [])
    ) or "none found"
    seg = r.get("segment", {})
    return (
        f"IDEA: {r['idea']}\n\n"
        f"Founder plan — team size, planned funding, and target market are baked "
        f"into the success model.\n\n"
        f"ML PIPELINE OUTPUT:\n"
        f"- Detected category: {r['category']} ({round(r['confidence']*100)}% confidence)\n"
        f"- Viability score: {r['score']}/100  ->  {r['verdict']}\n"
        f"- Success probability: {round(r['success_prob']*100)}%\n"
        f"- Market sentiment: {round(r['sentiment_pos']*100)}% positive\n"
        f"- Novelty: {round(r['axes']['Novelty'])}/100 (higher = less crowded)\n"
        f"- Category historical success base rate: {round(r['category_base']*100)}%\n"
        f"- Weakest axis: {r['weakest_axis']}\n"
        f"- Nearest existing ideas: {comps}\n"
        f"- Target segment (KMeans): {seg.get('label', 'n/a')}\n\n"
        f"Give your advice as the JSON object described in the system prompt."
    )


def _parse_json(text):
    """Best-effort JSON extraction from the model's text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _chat(cfg, system, user):
    """POST an OpenAI-style chat completion and return the assistant text."""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": 1200,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['key']}",
        # Some providers sit behind Cloudflare, which blocks the default
        # "Python-urllib" agent (403 / error 1010). A normal UA avoids that.
        "User-Agent": "Mozilla/5.0 (compatible; StartupAdvisorLM/1.0)",
    })
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def advise(report):
    """Return the advice dict, or {'available': False, 'reason': ...}."""
    ok, reason = smart_available()
    if not ok:
        return {"available": False, "reason": reason}

    cfg = _resolve()
    try:
        text = _chat(cfg, SYSTEM_PROMPT, _build_user_prompt(report))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            body = ""
        return {"available": False,
                "reason": f"{cfg['provider']} API error {e.code}. {body}".strip()}
    except Exception as e:  # network / timeout / bad response -> graceful
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    try:
        advice = _parse_json(text)
    except Exception:
        return {"available": False, "reason": "Model did not return valid JSON."}

    advice["available"] = True
    advice["model"] = f"{cfg['provider']}/{cfg['model']}"
    return advice


# Module-level display label used by /smart_status (env is read at server start).
_c = _resolve()
PROVIDER = _c["provider"]
MODEL = f"{_c['provider']}/{_c['model']}" if _c["model"] else _c["provider"]
