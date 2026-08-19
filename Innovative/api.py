"""
api.py — StartupAdvisorLM backend
---------------------------------
A tiny FastAPI service that serves the custom web frontend AND exposes the
existing ML pipeline (src/) over JSON. The data-science 'brain' is unchanged —
this is purely a nicer face on top of src/pipeline.py.

Run from the `startup/` folder:
    python generate_data.py          # once, to create the datasets
    uvicorn api:app --port 8000      # then open http://localhost:8000
"""

import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline import build_engine, validate
from src import advisor

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")

app = FastAPI(title="StartupAdvisorLM")

# Build every model once at startup (cached for the process lifetime)
engine = build_engine()


class IdeaRequest(BaseModel):
    idea: str
    team_size: int = 8
    funding_k: float = 150
    market_size: int = 6


def _to_native(o):
    """Let json handle numpy/pandas scalar types."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def _clean(obj):
    return json.loads(json.dumps(obj, default=_to_native))


@app.post("/validate")
def do_validate(req: IdeaRequest):
    """Run one idea through the full pipeline and return the report."""
    report = validate(engine, req.idea, req.team_size, req.funding_k, req.market_size)
    return JSONResponse(_clean(report))


@app.get("/smart_status")
def smart_status():
    """Tell the frontend whether LLM Smart Mode is usable."""
    ok, reason = advisor.smart_available()
    return JSONResponse({"available": ok, "reason": reason, "model": advisor.MODEL})


@app.post("/advise")
def do_advise(req: IdeaRequest):
    """Run the pipeline, then layer Claude's qualitative advice on top."""
    report = validate(engine, req.idea, req.team_size, req.funding_k, req.market_size)
    advice = advisor.advise(_clean(report))
    return JSONResponse(_clean(advice))


@app.get("/segments")
def get_segments():
    """Customer scatter data for the segmentation chart (fetched once on load)."""
    seg = engine["segmenter"]
    df = seg["data"]
    points = [{"income": int(r.income_k), "spend": int(r.spend_score),
               "segment": int(r.segment)} for r in df.itertuples()]
    return JSONResponse(_clean({
        "points": points,
        "descriptions": seg["descriptions"],
        "sizes": seg["sizes"],
    }))


# Serve the static frontend at the root (must be mounted AFTER the API routes)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
