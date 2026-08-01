"""FastAPI service for the Gift Recommendation System."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

# ── fix import path so gift_engine / text_block / exposure are always found ──
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # api/  (contains the copied modules)

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import exposure      # noqa: F401  required for joblib unpickling
import gift_engine   # noqa: F401  required for joblib unpickling
import text_block    # noqa: F401  required for joblib unpickling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gift_api")

# ── absolute paths so the server can be launched from any working directory ──
_ROOT         = _HERE.parent
ARTIFACT_PATH = _ROOT / "models" / "gift_recommender_v3.joblib"
OFFERS_PATH   = _ROOT / "data" / "processed" / "catalog_offers.csv"
EXPOSURE_ENABLED = False

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup, never per request."""
    t0 = time.perf_counter()

    logger.info("looking for artifact at %s", ARTIFACT_PATH)
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"model artifact not found at {ARTIFACT_PATH}")

    artifact = joblib.load(ARTIFACT_PATH)
    engine   = artifact["recommender"]

    engine.exposure = (
        exposure.ExposureTracker(len(engine.catalog), lam=1.0)
        if EXPOSURE_ENABLED else None
    )

    STATE["engine"]        = engine
    STATE["metadata"]      = artifact["metadata"]
    STATE["cluster_names"] = artifact.get("cluster_names", {})

    if OFFERS_PATH.exists():
        offers = pd.read_csv(OFFERS_PATH, encoding="utf-8-sig")
        STATE["offers"] = offers.set_index("parent_id").sort_index()
        logger.info("loaded %d purchase options", len(offers))
    else:
        STATE["offers"] = None
        logger.warning("%s not found — /product/.../options will return empty", OFFERS_PATH)

    logger.info("engine ready: %d products in %.1fs",
                len(engine.catalog), time.perf_counter() - t0)
    yield
    STATE.clear()


app = FastAPI(
    title="Gift Recommender API",
    version="3.0.0",
    description=(
        "Context-aware gift recommendation. Age and budget are hard "
        "constraints and are never violated."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── schemas ──────────────────────────────────────────────────────────────────
class GiftRequest(BaseModel):
    age:       int   = Field(..., ge=0, le=99)
    gender:    Literal["Female", "Male", "Any"] = "Any"
    occasion:  str
    budget:    float = Field(..., gt=0, le=100_000)
    interests: list[str] = Field(default_factory=list, max_length=10)
    top_k:     int  = Field(6, ge=1, le=24)

    @field_validator("interests")
    @classmethod
    def strip_blanks(cls, v: list[str]) -> list[str]:
        return [x.strip() for x in v if x and x.strip()]


# ── helpers ───────────────────────────────────────────────────────────────────
def build_reasons(row: pd.Series, req: GiftRequest) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("occasion_match")):
        reasons.append(f"Suited to {req.occasion}")
    if req.interests:
        reasons.append("Matches " + ", ".join(req.interests[:2]))
    share = row["price_min"] / req.budget if req.budget else 0
    reasons.append(f"Uses {share:.0%} of your budget")
    if int(row.get("offer_count", 1)) > 1:
        reasons.append(f"{int(row['offer_count'])} available options")
    return reasons


def to_rec(row: pd.Series, req: GiftRequest) -> dict[str, Any]:
    image = row.get("image_display_url") or row.get("image_url")
    if isinstance(image, str) and not image.startswith(("http://", "https://")):
        image = None
    return {
        "parent_id":           str(row["parent_id"]),
        "product_name":        str(row["product_name"]),
        "brand":               str(row["brand"]),
        "category":            str(row["category"]),
        "sub_category":        str(row["sub_category"]),
        "price_min":           float(row["price_min"]),
        "price_max":           float(row["price_max"]),
        "price_median":        float(row["price_median"]),
        "offer_count":         int(row["offer_count"]),
        "product_url":         str(row["product_url"]),
        "image_url":           image,
        "image_display_url":   image,
        "match_score":         round(float(row["match_score"]),   4),
        "interest_similarity": round(float(row["interest_similarity"]), 4),
        "occasion_match":      bool(row["occasion_match"]),
        "budget_fit":          round(float(row["budget_fit"]),    3),
        "reasons":             build_reasons(row, req),
    }


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "Gift Recommender API", "version": app.version,
            "docs": "/docs", "products": len(STATE["engine"].catalog)}


@app.get("/health")
def health() -> dict[str, Any]:
    ready = "engine" in STATE
    return {"status": "ok" if ready else "loading", "ready": ready}


@app.get("/options")
def options() -> dict[str, list[str]]:
    meta = STATE["metadata"]
    return {"interests": meta["interests"],
            "occasions": meta["occasions"],
            "genders":   meta["genders"]}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    meta = STATE["metadata"]
    return {"precision_at_5": meta.get("precision_at_5"),
            "ndcg_at_10":     meta.get("ndcg_at_10"),
            "n_items":        meta.get("n_items"),
            "n_features":     meta.get("n_features"),
            "clusters":       meta.get("k")}


@app.post("/recommend")
def recommend(req: GiftRequest) -> dict[str, Any]:
    engine, meta = STATE["engine"], STATE["metadata"]

    if req.occasion not in meta["occasions"]:
        raise HTTPException(422, f"Unknown occasion. Valid: {meta['occasions']}")

    unknown = set(req.interests) - set(meta["interests"])
    if unknown:
        raise HTTPException(422, f"Unknown interests {sorted(unknown)}")

    t0 = time.perf_counter()
    try:
        recs, diag = engine.recommend(
            age=req.age, gender=req.gender, occasion=req.occasion,
            budget=req.budget, interests=req.interests, top_k=req.top_k,
        )
    except Exception:
        logger.exception("recommendation failed")
        raise HTTPException(500, "Recommendation engine error") from None

    elapsed = (time.perf_counter() - t0) * 1000
    items   = [to_rec(row, req) for _, row in recs.iterrows()]

    breaches = [i["parent_id"] for i in items if i["price_min"] > req.budget]
    if breaches:
        logger.error("BUDGET VIOLATION budget=%s items=%s", req.budget, breaches)
        raise HTTPException(500, "Internal consistency error: budget violated")

    logger.info("recommend age=%s occasion=%s budget=%s -> %d in %.0fms",
                req.age, req.occasion, req.budget, len(items), elapsed)

    return {
        "segment":         STATE["cluster_names"].get(diag["cluster_id"], "General gifts"),
        "query":           req.model_dump(),
        "diagnostics":     {**diag, "latency_ms": round(elapsed, 1),
                            "n_returned": len(items)},
        "recommendations": items,
    }


@app.get("/product/{parent_id}/options")
def product_options(parent_id: str) -> dict[str, Any]:
    offers = STATE.get("offers")
    if offers is None:
        raise HTTPException(503, "Offers table not loaded")
    if parent_id not in offers.index:
        raise HTTPException(404, f"Unknown product '{parent_id}'")
    rows = offers.loc[[parent_id]].sort_values("price")
    return {
        "parent_id":    parent_id,
        "option_count": len(rows),
        "options": [
            {"offer_id":    str(r["offer_id"]),
             "price":       float(r["price"]),
             "product_url": str(r["Product URL"]),
             "image_url":   str(r["Image URL"]) if pd.notna(r.get("Image URL")) else None,
             "source_site": str(r["Source Site"])}
            for _, r in rows.iterrows()
        ],
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
