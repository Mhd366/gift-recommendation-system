"""Gift recommendation engine: hard filtering, weighted scoring, diverse ranking.

Three stages, deliberately separated:

1. **Hard filters** - budget, age, availability. Binary and non-negotiable. An
   item outside the budget is removed, never merely down-weighted.
2. **Weighted scoring** - occasion, interest, category, gender and price fit,
   combined into ``[0, 1]``. Occasion and interest matches are weighted by
   inverse label frequency, so a rare match (Housewarming, Fragrance) counts for
   more than a near-universal one (Birthday, Eid).
3. **Diverse re-ranking** - Maximal Marginal Relevance with a per-category cap,
   so the list is not five variations of one product.

Usage::

    from gift_recommender.recommendation.engine import GiftRecommender, Query

    engine = GiftRecommender.from_files(
        "data/processed/catalog_clean.csv",
        "data/processed/signal_weights.csv",
    )
    results = engine.recommend(Query(age=25, budget=400, occasion="Graduation",
                                    interests=["Technology"]))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "occasion": 0.32,
    "interest": 0.30,
    "category": 0.13,
    "price_fit": 0.15,
    "gender": 0.05,
    "quality": 0.05,
}


@dataclass
class Query:
    """A recommendation request describing the gift recipient."""

    age: int
    budget: float
    occasion: str | None = None
    interests: list[str] = field(default_factory=list)
    category: str | None = None
    gender: str | None = None
    top_k: int = 10
    budget_floor_ratio: float = 0.25


@dataclass
class Recommendation:
    """A single scored result."""

    parent_id: str
    product_name: str
    brand: str
    category: str
    price_min: float
    price_max: float
    offer_count: int
    score: float
    reasons: list[str]
    product_url: str
    image_url: str


class GiftRecommender:
    """Content-based recommender over the cleaned product catalogue."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        weights: pd.DataFrame,
        score_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialise the recommender.

        Args:
            catalog: Cleaned product table, one row per parent product.
            weights: Signal weight table indexed by label, with ``axis`` and
                ``idf_weight`` columns.
            score_weights: Component weights. Must sum to 1.0.
        """
        self.catalog = catalog.reset_index(drop=True)
        self.score_weights = score_weights or DEFAULT_WEIGHTS

        total = sum(self.score_weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"score weights must sum to 1.0, got {total:.4f}")

        self.occasion_idf = weights.loc[weights["axis"] == "occasion", "idf_weight"].to_dict()
        self.interest_idf = weights.loc[weights["axis"] == "interest", "idf_weight"].to_dict()
        self._max_occ_idf = max(self.occasion_idf.values())
        self._max_int_idf = max(self.interest_idf.values())

    @classmethod
    def from_files(cls, catalog_path: str | Path, weights_path: str | Path) -> GiftRecommender:
        """Build a recommender from the pipeline's CSV outputs."""
        catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
        weights = pd.read_csv(weights_path, encoding="utf-8-sig", index_col="label")
        return cls(catalog, weights)

    # -- stage 1 -----------------------------------------------------------
    def _hard_filter(self, query: Query) -> pd.DataFrame:
        """Apply non-negotiable constraints, relaxing only the budget floor.

        Budget and age are never relaxed. If the result set is empty the caller
        receives an empty frame rather than a violating recommendation.
        """
        df = self.catalog
        mask = (
            (df["price_min"] <= query.budget)
            & (df["min_age"] <= query.age)
            & (df["max_age"] >= query.age)
        )
        candidates = df[mask]
        logger.debug("hard filter: %d -> %d candidates", len(df), len(candidates))
        return candidates

    # -- stage 2 -----------------------------------------------------------
    def _score(self, candidates: pd.DataFrame, query: Query) -> pd.DataFrame:
        """Compute a weighted relevance score for every candidate."""
        scored = candidates.copy()
        n = len(scored)

        # Occasion: an IDF-weighted match, normalised so rare labels dominate.
        if query.occasion and f"occ_{query.occasion}" in scored.columns:
            hit = scored[f"occ_{query.occasion}"].astype(bool).to_numpy()
            idf = self.occasion_idf.get(query.occasion, 1.0)
            occasion = hit * (idf / self._max_occ_idf)
        else:
            occasion = np.full(n, 0.5)

        # Interest: mean IDF-weighted coverage of the requested interests.
        requested = [i for i in query.interests if f"int_{i}" in scored.columns]
        if requested:
            contributions = [
                scored[f"int_{i}"].astype(bool).to_numpy()
                * (self.interest_idf.get(i, 1.0) / self._max_int_idf)
                for i in requested
            ]
            interest = np.mean(contributions, axis=0)
        else:
            interest = np.full(n, 0.5)

        category = (
            (scored["category"] == query.category).astype(float).to_numpy()
            if query.category else np.full(n, 0.5)
        )

        # Price fit: reward using the budget well. A gift at 15% of budget reads
        # as cheap; one near the ceiling reads as considered.
        ratio = (scored["price_min"] / query.budget).clip(0, 1).to_numpy()
        floor = query.budget_floor_ratio
        price_fit = np.where(ratio < floor, ratio / floor, 1.0 - (ratio - floor) * 0.3)

        if query.gender:
            gender = np.where(
                scored["gender_target"].to_numpy() == query.gender, 1.0,
                np.where(scored["gender_target"].to_numpy() == "Unisex", 0.7, 0.3),
            )
        else:
            gender = np.full(n, 0.5)

        quality = (
            scored["has_image"].astype(float).to_numpy() * 0.7
            + (scored["offer_count"] > 1).astype(float).to_numpy() * 0.3
        )

        w = self.score_weights
        scored["score"] = (
            w["occasion"] * occasion
            + w["interest"] * interest
            + w["category"] * category
            + w["price_fit"] * price_fit
            + w["gender"] * gender
            + w["quality"] * quality
        )
        scored["_occasion"] = occasion
        scored["_interest"] = interest
        return scored.sort_values("score", ascending=False)

    # -- stage 3 -----------------------------------------------------------
    def _rerank(
        self,
        scored: pd.DataFrame,
        top_k: int,
        lambda_: float = 0.7,
        max_per_category: int = 3,
    ) -> pd.DataFrame:
        """Re-rank with Maximal Marginal Relevance and a per-category cap.

        Args:
            scored: Candidates sorted by score.
            top_k: Number of results to return.
            lambda_: 1.0 is pure relevance, 0.0 is pure diversity.
            max_per_category: Hard cap on results from any one category.
        """
        pool = scored.head(min(len(scored), top_k * 30))
        selected: list[int] = []
        category_counts: dict[str, int] = {}

        for _ in range(min(top_k, len(pool))):
            best_idx, best_value = None, -np.inf
            for idx, row in pool.iterrows():
                if idx in selected:
                    continue
                if category_counts.get(row["category"], 0) >= max_per_category:
                    continue
                penalty = sum(
                    1.0 for s in selected
                    if pool.loc[s, "category"] == row["category"]
                    or pool.loc[s, "brand"] == row["brand"]
                )
                value = lambda_ * row["score"] - (1 - lambda_) * penalty * 0.1
                if value > best_value:
                    best_idx, best_value = idx, value
            if best_idx is None:
                break
            selected.append(best_idx)
            category_counts[pool.loc[best_idx, "category"]] = (
                category_counts.get(pool.loc[best_idx, "category"], 0) + 1
            )

        return pool.loc[selected]

    # -- explanation -------------------------------------------------------
    def _reasons(self, row: pd.Series, query: Query) -> list[str]:
        """Build human-readable justifications shown on the product card."""
        out: list[str] = []
        if query.occasion and row.get(f"occ_{query.occasion}", False):
            out.append(f"Suitable for {query.occasion}")
        matched = [i for i in query.interests if row.get(f"int_{i}", False)]
        if matched:
            out.append("Matches interest: " + ", ".join(matched))
        if row["price_min"] <= query.budget:
            share = row["price_min"] / query.budget
            out.append(f"Within budget ({share:.0%} of it)")
        if row["offer_count"] > 1:
            out.append(f"{int(row['offer_count'])} available options")
        return out

    def recommend(self, query: Query) -> list[Recommendation]:
        """Return the top-K recommendations for a query.

        Returns an empty list rather than a constraint-violating result when no
        candidate satisfies the hard filters.
        """
        candidates = self._hard_filter(query)
        if candidates.empty:
            logger.warning("no candidates satisfy the hard constraints")
            return []

        ranked = self._rerank(self._score(candidates, query), query.top_k)
        return [
            Recommendation(
                parent_id=row["parent_id"],
                product_name=row["product_name"],
                brand=row["brand"],
                category=row["category"],
                price_min=float(row["price_min"]),
                price_max=float(row["price_max"]),
                offer_count=int(row["offer_count"]),
                score=round(float(row["score"]), 4),
                reasons=self._reasons(row, query),
                product_url=row["product_url"],
                image_url=row["image_url"],
            )
            for _, row in ranked.iterrows()
        ]
