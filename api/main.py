"""FastAPI service for the Gift Recommendation System.

Loads the serialised retrieval engine once at import and serves ranked gift
recommendations. The engine is stateless and read-only after load, so it is
safe to share across uvicorn workers — with one exception noted below.

Deployment requirements
-----------------------
``gift_engine.py``, ``text_block.py`` and ``exposure.py`` must be importable.
joblib pickles reference classes by module path, so without them ``joblib.load``
raises ``AttributeError``.

Pin scikit-learn to the version recorded in the artifact metadata. Unpickling
estimators across major versions is not guaranteed.

Exposure damping keeps an in-process counter, so each worker damps
independently. With several workers the effect is weaker but still positive.
For balanced exposure across a real deployment the counter needs shared state
(Redis or equivalent) — see ``EXPOSURE_ENABLED``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import exposure  # noqa: F401  - required for unpickling
import gift_engine  # noqa: F401  - required for unpickling
import text_block  # noqa: F401  - required for unpickling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gift_api")

ARTIFACT_PATH = Path("models/gift_recommender_v3.joblib")
OFFERS_PATH = Path("data/processed/catalog_offers.csv")
EXPOSURE_ENABLED = False  # single-worker only; needs shared state otherwise

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup, never per request."""
    t0 = time.perf_counter()
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"model artifact not found at {ARTIFACT_PATH}")

    artifact = joblib.load(ARTIFACT_PATH)
    engine = artifact["recommender"]

    if EXPOSURE_ENABLED:
        engine.exposure = exposure.ExposureTracker(len(engine.catalog), lam=1.0)
    else:
        engine.exposure = None

    STATE["engine"] = engine
    STATE["metadata"] = artifact["metadata"]
    STATE["cluster_names"] = artifact.get("cluster_names", {})

    # Purchase options are looked up per product, so an index makes it O(1).
    if OFFERS_PATH.exists():
        offers = pd.read_csv(OFFERS_PATH, encoding="utf-8-sig")
        STATE["offers"] = offers.set_index("parent_id").sort_index()
        logger.info("loaded %d purchase options", len(offers))
    else:
        STATE["offers"] = None
        logger.warning("%s not found - the options endpoint will return empty",
                       OFFERS_PATH)

    logger.info("engine ready: %d products, %.1fs",
                len(engine.catalog), time.perf_counter() - t0)
    yield
    STATE.clear()


app = FastAPI(
    title="Gift Recommender API",
    version="3.0.0",
    description=(
        "Context-aware gift recommendation. Age and budget are hard "
        "constraints and are never violated; occasion, interest, gender and "
        "price fit are weighted scores."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before a public deployment
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class GiftRequest(BaseModel):
    """A recommendation request describing the gift recipient."""

    age: int = Field(..., ge=0, le=99, description="Recipient age in years")
    gender: Literal["Female", "Male", "Any"] = Field(
        "Any", description="Soft signal only; never used to exclude products")
    occasion: str = Field(..., description="One of the occasions from /options")
    budget: float = Field(..., gt=0, le=100_000, description="Ceiling in SAR")
    interests: list[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(6, ge=1, le=24)

    @field_validator("interests")
    @classmethod
    def strip_blanks(cls, value: list[str]) -> list[str]:
        """Drop empty entries so a stray blank does not weaken the query."""
        return [v.strip() for v in value if v and v.strip()]


class Recommendation(BaseModel):
    """One ranked gift."""

    parent_id: str
    product_name: str
    brand: str
    category: str
    sub_category: str
    price_min: float
    price_max: float
    price_median: float
    offer_count: int
    product_url: str
    image_url: str | None = None
    match_score: float
    interest_similarity: float
    occasion_match: bool
    budget_fit: float
    reasons: list[str]


class RecommendResponse(BaseModel):
    """A full recommendation response."""

    segment: str
    query: dict[str, Any]
    diagnostics: dict[str, Any]
    recommendations: list[Recommendation]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_reasons(row: pd.Series, req: GiftRequest) -> list[str]:
    """Explain, in the user's terms, why this product was surfaced."""
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


def to_recommendation(row: pd.Series, req: GiftRequest) -> dict[str, Any]:
    """Convert an engine row into the API response shape."""
    image = row.get("image_display_url") or row.get("image_url")
    if isinstance(image, str) and not image.startswith(("http://", "https://")):
        image = None
    return {
        "parent_id": str(row["parent_id"]),
        "product_name": str(row["product_name"]),
        "brand": str(row["brand"]),
        "category": str(row["category"]),
        "sub_category": str(row["sub_category"]),
        "price_min": float(row["price_min"]),
        "price_max": float(row["price_max"]),
        "price_median": float(row["price_median"]),
        "offer_count": int(row["offer_count"]),
        "product_url": str(row["product_url"]),
        "image_url": image,
        "match_score": round(float(row["match_score"]), 4),
        "interest_similarity": round(float(row["interest_similarity"]), 4),
        "occasion_match": bool(row["occasion_match"]),
        "budget_fit": round(float(row["budget_fit"]), 3),
        "reasons": build_reasons(row, req),
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    """Service banner."""
    return {
        "service": "Gift Recommender API",
        "version": app.version,
        "docs": "/docs",
        "products": len(STATE["engine"].catalog),
    }


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness probe for a load balancer or container orchestrator."""
    ready = "engine" in STATE
    return {"status": "ok" if ready else "loading", "ready": ready}


@app.get("/options", tags=["meta"])
def options() -> dict[str, list[str]]:
    """Valid dropdown values, so the UI never sends an unknown label."""
    meta = STATE["metadata"]
    return {
        "interests": meta["interests"],
        "occasions": meta["occasions"],
        "genders": meta["genders"],
    }


@app.get("/metrics", tags=["meta"])
def metrics() -> dict[str, Any]:
    """Offline evaluation figures baked into the artifact at training time."""
    meta = STATE["metadata"]
    return {
        "precision_at_5": meta.get("precision_at_5"),
        "ndcg_at_10": meta.get("ndcg_at_10"),
        "n_items": meta.get("n_items"),
        "n_features": meta.get("n_features"),
        "clusters": meta.get("k"),
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: GiftRequest) -> dict[str, Any]:
    """Return the top-K gifts for a recipient profile.

    Age and budget are hard filters: no returned product ever exceeds the
    stated budget or falls outside the recipient's age range. An empty result
    means no product in the catalogue satisfies those constraints, which is a
    supply gap rather than a ranking failure.
    """
    engine, meta = STATE["engine"], STATE["metadata"]

    if req.occasion not in meta["occasions"]:
        raise HTTPException(
            422, f"Unknown occasion '{req.occasion}'. Valid: {meta['occasions']}")

    unknown = set(req.interests) - set(meta["interests"])
    if unknown:
        raise HTTPException(
            422, f"Unknown interests {sorted(unknown)}. Valid: {meta['interests']}")

    t0 = time.perf_counter()
    try:
        recs, diag = engine.recommend(
            age=req.age, gender=req.gender, occasion=req.occasion,
            budget=req.budget, interests=req.interests, top_k=req.top_k,
        )
    except Exception:
        logger.exception("recommendation failed for %s", req.model_dump())
        raise HTTPException(500, "Recommendation engine error") from None

    elapsed = (time.perf_counter() - t0) * 1000
    items = [to_recommendation(row, req) for _, row in recs.iterrows()]

    # A hard constraint breach is a bug, not a degraded result. Fail loudly.
    breaches = [i["parent_id"] for i in items if i["price_min"] > req.budget]
    if breaches:
        logger.error("BUDGET VIOLATION budget=%s items=%s", req.budget, breaches)
        raise HTTPException(500, "Internal consistency error: budget constraint violated")

    logger.info("recommend age=%s occasion=%s budget=%s -> %d results in %.0fms",
                req.age, req.occasion, req.budget, len(items), elapsed)

    return {
        "segment": STATE["cluster_names"].get(diag["cluster_id"], "General gifts"),
        "query": req.model_dump(),
        "diagnostics": {**diag, "latency_ms": round(elapsed, 1),
                        "n_returned": len(items)},
        "recommendations": items,
    }


@app.get("/product/{parent_id}/options", tags=["recommend"])
def product_options(parent_id: str) -> dict[str, Any]:
    """List every purchasable variant of a product.

    The recommender ranks parent products so one item appears once, and this
    endpoint resolves the chosen product to its colour, size and bundle
    variants for the product page.
    """
    offers = STATE.get("offers")
    if offers is None:
        raise HTTPException(503, "Offers table not loaded")
    if parent_id not in offers.index:
        raise HTTPException(404, f"Unknown product '{parent_id}'")

    rows = offers.loc[[parent_id]].sort_values("price")
    return {
        "parent_id": parent_id,
        "option_count": len(rows),
        "options": [
            {
                "offer_id": str(r["offer_id"]),
                "price": float(r["price"]),
                "product_url": str(r["Product URL"]),
                "image_url": (str(r["Image URL"])
                              if pd.notna(r.get("Image URL")) else None),
                "source_site": str(r["Source Site"]),
            }
            for _, r in rows.iterrows()
        ],
    }


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean payload instead of a stack trace on an unexpected error."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
