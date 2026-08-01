"""Feature engineering and retrieval logic for the Gift Recommendation System.

Imported by both the training notebook and the FastAPI service. joblib pickles
reference functions by module path, so logic defined only inside a notebook
(``__main__``) fails to unpickle in the API process. Keeping it here makes the
artefact portable and guarantees train/serve parity.

Changes from v1, each driven by a measured defect:

* ``budget_tolerance`` defaults to **0.0**. v1 shipped 0.10, so a 200 SAR
  request could return a 220 SAR item while the evaluation asserted against
  ``budget * 1.10`` — the same expression the filter used. That assertion was
  tautological and could not fail.
* Occasion and gender are **scored, not filtered**. As filters they emptied
  results for rare occasions and hard-coded a gender stereotype into
  admissibility.
* Occasion and interest dimensions are **weighted by inverse label frequency**.
  A label covering 98% of the catalogue carries no ranking information;
  without weighting it competed on equal terms with one covering 6%.
* Score ties are broken by a **query-seeded jitter**. v1's deterministic prior
  made only the single prior-winning item of each duplicate profile reachable,
  capping catalogue coverage at 4%.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. CANONICAL VOCABULARIES (single source of truth for catalog and query)
# --------------------------------------------------------------------------
INTEREST_FLAGS = [
    "int_Art & Creativity", "int_Beauty & Grooming", "int_Cooking & Food",
    "int_Fashion & Style", "int_Fragrance", "int_Gaming",
    "int_Gardening & Nature", "int_Home & Interiors", "int_Jewellery & Watches",
    "int_Kids & Play", "int_Outdoors & Travel", "int_Reading & Learning",
    "int_Sports & Fitness", "int_Technology", "int_Wellness",
]
OCCASION_FLAGS = [
    "occ_Birthday", "occ_Graduation", "occ_Anniversary", "occ_Wedding",
    "occ_Eid", "occ_MothersDay", "occ_FathersDay", "occ_NewBaby",
    "occ_Housewarming", "occ_ThankYou",
]
INTERESTS = [c[len("int_"):] for c in INTEREST_FLAGS]
OCCASIONS = [c[len("occ_"):] for c in OCCASION_FLAGS]

AGE_BUCKETS = [
    ("age_baby", 0, 3), ("age_child", 4, 12), ("age_teen", 13, 17),
    ("age_youngadult", 18, 29), ("age_adult", 30, 49), ("age_senior", 50, 99),
]
AGE_BUCKET_COLS = [b[0] for b in AGE_BUCKETS]

PRICE_BANDS = ["Under 100 SAR", "100-299 SAR", "300-699 SAR",
               "700-1499 SAR", "1500+ SAR"]
PRICE_BAND_COLS = [f"band_{i}" for i in range(len(PRICE_BANDS))]
_BAND_EDGES = [100, 300, 700, 1500]

GENDER_COLS = ["serves_female", "serves_male"]
CATEGORICAL_COLS = ["category", "gift_type"]

NUMERIC_BLOCKS = {
    "interests": INTEREST_FLAGS,
    "occasions": OCCASION_FLAGS,
    "age": AGE_BUCKET_COLS,
    "gender": GENDER_COLS,
    "price": PRICE_BAND_COLS,
}
ALL_FEATURE_COLS = (INTEREST_FLAGS + OCCASION_FLAGS + AGE_BUCKET_COLS
                    + GENDER_COLS + PRICE_BAND_COLS + CATEGORICAL_COLS)


# --------------------------------------------------------------------------
# 2. LABEL WEIGHTING
# --------------------------------------------------------------------------
def load_signal_weights(path: str = "signal_weights.csv") -> dict[str, float]:
    """Load inverse-frequency weights, normalised so the maximum is 1.0.

    A label present on nearly the whole catalogue cannot separate one candidate
    from another. Weighting each dimension by log(1/frequency) makes a rare
    match (Housewarming, 5.7%) count for far more than a near-universal one
    (Birthday, 98.4%), which is what turns the occasion block into a genuine
    ranking signal rather than a constant.

    Falls back to uniform weights when the file is absent.
    """
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", index_col="label")["idf_weight"]
    except (FileNotFoundError, KeyError):
        return {c: 1.0 for c in OCCASION_FLAGS + INTEREST_FLAGS}

    weights: dict[str, float] = {}
    for flags, names in ((OCCASION_FLAGS, OCCASIONS), (INTEREST_FLAGS, INTERESTS)):
        vals = np.array([raw.get(n, 1.0) for n in names], dtype=float)
        vals = vals / vals.max()
        weights.update(dict(zip(flags, vals, strict=True)))
    return weights


# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def _price_to_band_index(price: float) -> int:
    """Map an absolute price to its ordinal band index."""
    return int(np.searchsorted(_BAND_EDGES, price, side="right"))


def _smoothed_band_vector(band_idx: int, neighbour_weight: float = 0.5) -> np.ndarray:
    """One-hot on the price band, softened onto the two adjacent bands.

    A hard one-hot makes a 299 SAR and a 301 SAR gift orthogonal, which is
    economically meaningless. Bleeding weight into neighbours encodes
    ordinality while keeping every entry non-negative.
    """
    v = np.zeros(len(PRICE_BANDS), dtype=np.float32)
    v[band_idx] = 1.0
    if band_idx - 1 >= 0:
        v[band_idx - 1] = neighbour_weight
    if band_idx + 1 < len(PRICE_BANDS):
        v[band_idx + 1] = neighbour_weight
    return v


def build_catalog_profile(df: pd.DataFrame,
                          weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Turn the cleaned catalogue into the shared feature frame.

    Every block is query-derivable, so one fitted preprocessor serves both
    catalogue and query.
    """
    weights = weights or {}
    out = pd.DataFrame(index=df.index)

    for col in INTEREST_FLAGS + OCCASION_FLAGS:
        out[col] = df[col].astype(np.float32) * weights.get(col, 1.0)

    # Age: an item flags every life stage its [min_age, max_age] interval
    # covers; a query flags only its own stage. This makes a point-vs-interval
    # comparison symmetric, and mildly prefers targeted gifts over 0-99 ones.
    lo, hi = df["min_age"].to_numpy(), df["max_age"].to_numpy()
    for name, b_lo, b_hi in AGE_BUCKETS:
        out[name] = ((lo <= b_hi) & (hi >= b_lo)).astype(np.float32)

    # Gender: "who does this serve", not a 3-way one-hot. Unisex -> [1,1] is a
    # genuine partial match for a gendered query rather than a third class.
    g = df["gender_target"].to_numpy()
    out["serves_female"] = np.isin(g, ["Female", "Unisex"]).astype(np.float32)
    out["serves_male"] = np.isin(g, ["Male", "Unisex"]).astype(np.float32)

    band_idx = df["price_median"].map(_price_to_band_index).to_numpy()
    out[PRICE_BAND_COLS] = np.vstack([_smoothed_band_vector(i) for i in band_idx])

    out["category"] = df["category"].to_numpy()
    out["gift_type"] = df["gift_type"].to_numpy()
    return out[ALL_FEATURE_COLS]


def build_interest_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Most probable taxonomy per interest, from catalogue co-occurrence.

    A user query carries no category or gift_type. Inferring them from the
    stated interests lets the taxonomy block do real work instead of sitting
    empty. This is a marginal frequency table, not a learned label.
    """
    rows = {}
    for flag, name in zip(INTEREST_FLAGS, INTERESTS, strict=True):
        sub = df.loc[df[flag].astype(bool)]
        if len(sub):
            rows[name] = {"category": sub["category"].mode().iat[0],
                          "gift_type": sub["gift_type"].mode().iat[0]}
    return pd.DataFrame.from_dict(rows, orient="index")


def build_query_profile(age: int, gender: str, occasion: str, budget: float,
                        interests: list[str],
                        affinity: pd.DataFrame | None = None,
                        weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Project a user request into the same feature space as the catalogue."""
    weights = weights or {}
    row = {c: 0.0 for c in ALL_FEATURE_COLS if c not in CATEGORICAL_COLS}

    for it in interests or []:
        key = f"int_{it}"
        if key in row:
            row[key] = weights.get(key, 1.0)

    okey = f"occ_{occasion}"
    if okey in row:
        row[okey] = weights.get(okey, 1.0)

    for name, b_lo, b_hi in AGE_BUCKETS:
        if b_lo <= age <= b_hi:
            row[name] = 1.0
            break

    if gender == "Female":
        row["serves_female"] = 1.0
    elif gender == "Male":
        row["serves_male"] = 1.0
    else:
        row["serves_female"] = row["serves_male"] = 1.0

    row.update(dict(zip(PRICE_BAND_COLS,
                        _smoothed_band_vector(_price_to_band_index(budget)),
                        strict=True)))

    q = pd.DataFrame([row])
    cat = gtype = "__UNKNOWN__"
    if affinity is not None and interests:
        hits = affinity.reindex(interests).dropna()
        if len(hits):
            cat = hits["category"].mode().iat[0]
            gtype = hits["gift_type"].mode().iat[0]
    q["category"], q["gift_type"] = cat, gtype
    return q[ALL_FEATURE_COLS]


# --------------------------------------------------------------------------
# 4. HARD CONSTRAINTS
# --------------------------------------------------------------------------
def apply_hard_filters(df: pd.DataFrame, age: int, budget: float,
                       budget_tolerance: float = 0.0) -> np.ndarray:
    """Boolean mask of catalogue rows admissible for this request.

    Only **age and budget** are enforced here. These are correctness
    constraints: a 900 SAR watch is not a slightly worse answer for a 200 SAR
    budget, it is a wrong one.

    Occasion and gender were filters in v1 and are now scored instead.
    Occasion as a filter emptied results for rare occasions — Housewarming
    covers 5.7% of the catalogue, and combined with age and budget it left as
    few as 397 candidates. Gender as a filter encodes a stereotype into
    admissibility for no measured relevance gain.

    ``budget_tolerance`` defaults to 0.0. Any non-zero value must be surfaced
    to the user as an explicit choice, never applied silently.
    """
    ceiling = budget * (1.0 + budget_tolerance)
    return (
        (df["min_age"].to_numpy() <= age)
        & (df["max_age"].to_numpy() >= age)
        & (df["price_min"].to_numpy() <= ceiling)
    )


def price_fit_score(price_min: np.ndarray, price_max: np.ndarray,
                    budget: float) -> np.ndarray:
    """How well a product's price range absorbs the stated budget.

    The catalogue stores a span per parent product, so a scalar price is the
    wrong object to compare. The shopper's sweet spot is taken as 85% of
    budget, projected onto the attainable range, and scored on the log gap.
    """
    target = 0.85 * budget
    attainable = np.clip(target, price_min, price_max)
    return np.exp(-1.5 * np.abs(np.log((attainable + 1e-9) / (target + 1e-9))))


def quality_prior(df: pd.DataFrame) -> np.ndarray:
    """Small prior favouring products that render well and are trustworthy."""
    conf = df["subcategory_confidence"].map(
        {"confirmed": 1.0, "unverifiable": 0.5, "conflicting": 0.0}
    ).fillna(0.5).to_numpy()
    return 0.7 * conf + 0.3 * df["has_image"].to_numpy().astype(float)


# --------------------------------------------------------------------------
# 5. THE RECOMMENDER
# --------------------------------------------------------------------------
class GiftRecommender:
    """Filter, score, diversify retrieval engine."""

    def __init__(self, preprocessor, kmeans, embeddings, catalog, affinity,
                 weights=None, w_sim=0.55, w_occasion=0.15, w_gender=0.05,
                 w_price=0.20, w_prior=0.05, budget_tolerance=0.0,
                 tie_jitter=0.05, text_pipeline=None, w_structural=0.70,
                 w_text=0.30, exposure=None):
        """Initialise the engine.

        Args:
            weights: Inverse-frequency label weights.
            w_occasion: Weight of the occasion match, now a score not a filter.
            w_gender: Deliberately small. Gender is a soft signal; as a filter
                it encodes stereotypes and shrinks the candidate pool.
            budget_tolerance: 0.0 by default. Never raise it silently.
            tie_jitter: Amplitude of the query-seeded tie-break. Thousands of
                products share an identical profile, so a deterministic prior
                made only one item per profile reachable and capped coverage at
                4%. Seeded jitter rotates exposure across tied items while
                keeping each query reproducible.
            text_pipeline: Fitted char-n-gram TF-IDF to SVD pipeline over
                product names. Without it 45,055 products collapse onto 2,203
                distinct vectors and 95% of the catalogue is geometrically
                indistinguishable. Supplying it raises that to 44,811.
            w_structural: Weight of the structural block in the combined space.
            w_text: Weight of the text block. Held below w_structural so text
                breaks ties inside a structurally-matched group rather than
                overriding constraint-derived signals.
        """
        self.preprocessor = preprocessor
        self.kmeans = kmeans
        self.embeddings = embeddings.astype(np.float32)
        self.catalog = catalog.reset_index(drop=True)
        self.affinity = affinity
        self.weights = weights or {}
        self.w_sim, self.w_occasion, self.w_gender = w_sim, w_occasion, w_gender
        self.w_price, self.w_prior = w_price, w_prior
        self.budget_tolerance = budget_tolerance
        self.tie_jitter = tie_jitter
        self.text_pipeline = text_pipeline
        self.w_structural, self.w_text = w_structural, w_text
        self.exposure = exposure
        self.labels_ = kmeans.labels_
        self._prior = quality_prior(self.catalog)
        self._price_min = self.catalog["price_min"].to_numpy()
        self._price_max = self.catalog["price_max"].to_numpy()
        self._subcat = self.catalog["sub_category"].to_numpy()
        self._gender = self.catalog["gender_target"].to_numpy()
        self._occ = {o: self.catalog[f"occ_{o}"].to_numpy().astype(bool)
                     for o in OCCASIONS if f"occ_{o}" in self.catalog.columns}

    @staticmethod
    def _seed(age, gender, occasion, budget, interests) -> int:
        """Deterministic seed from the query, so results are reproducible."""
        key = f"{age}|{gender}|{occasion}|{budget}|{sorted(interests or [])}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

    def _embed_query(self, age, gender, occasion, budget, interests):
        """Embed a query in the same space as the catalogue.

        When a text pipeline is present the query's text side is synthesised
        from its interests and occasion, so it lands in the combined space
        rather than the structural one alone.
        """
        raw = build_query_profile(age, gender, occasion, budget, interests,
                                  affinity=self.affinity, weights=self.weights)
        struct = self.preprocessor.transform(raw).astype(np.float32)
        if self.text_pipeline is None:
            return struct

        from text_block import normalise_name
        hint = " ".join((interests or []) + [occasion])
        text = self.text_pipeline.transform([normalise_name(hint)]).astype(np.float32)
        combined = np.hstack([struct * self.w_structural, text * self.w_text])
        norm = np.linalg.norm(combined, axis=1, keepdims=True)
        return (combined / np.clip(norm, 1e-9, None)).astype(np.float32)

    def _diversify(self, order, idx, top_k, max_per_subcategory):
        """Cap items per sub-category, back-filling if the cap starves the list."""
        picked, seen = [], {}
        for o in order:
            sc = self._subcat[idx[o]]
            if seen.get(sc, 0) >= max_per_subcategory:
                continue
            seen[sc] = seen.get(sc, 0) + 1
            picked.append(o)
            if len(picked) == top_k:
                return picked
        for o in order:
            if o not in picked:
                picked.append(o)
                if len(picked) == top_k:
                    break
        return picked

    def recommend(self, age, gender, occasion, budget, interests, top_k=5,
                  use_cluster_gate=False, max_per_subcategory=2):
        """Return ``(recommendations, diagnostics)`` for one request.

        Age and budget are enforced as filters; occasion, gender, interest and
        price fit are scored. The result set is never empty unless no product
        satisfies age and budget, which is a genuine supply gap.
        """
        interests = interests or []
        q = self._embed_query(age, gender, occasion, budget, interests)
        cluster_id = (int(self.kmeans.predict(q[:, : self.kmeans.n_features_in_])[0])
                      if q.shape[1] >= self.kmeans.n_features_in_ else -1)

        mask = apply_hard_filters(self.catalog, age, budget, self.budget_tolerance)
        n_admissible = int(mask.sum())

        if use_cluster_gate:
            gated = mask & (self.labels_ == cluster_id)
            if gated.sum() >= 5 * top_k:
                mask = gated

        idx = np.flatnonzero(mask)
        diag = {"cluster_id": cluster_id, "n_admissible": n_admissible,
                "n_scored": int(len(idx)), "cluster_gate": use_cluster_gate}
        if len(idx) == 0:
            return pd.DataFrame(), diag

        sim = self.embeddings[idx] @ q.ravel()

        occ_hit = (self._occ[occasion][idx].astype(float)
                   if occasion in self._occ else np.zeros(len(idx)))
        occ_w = self.weights.get(f"occ_{occasion}", 1.0)
        occ_score = occ_hit * occ_w

        if gender in ("Female", "Male"):
            g = self._gender[idx]
            gender_score = np.where(g == gender, 1.0, np.where(g == "Unisex", 0.7, 0.2))
        else:
            gender_score = np.full(len(idx), 0.7)

        pfit = price_fit_score(self._price_min[idx], self._price_max[idx], budget)

        rng = np.random.default_rng(self._seed(age, gender, occasion, budget, interests))
        jitter = rng.random(len(idx)) * self.tie_jitter

        score = (self.w_sim * sim + self.w_occasion * occ_score
                 + self.w_gender * gender_score + self.w_price * pfit
                 + self.w_prior * self._prior[idx] + jitter)

        # Damp items that have already been shown many times, so the long tail
        # gets exposure instead of the same winners recurring on every query.
        if self.exposure is not None:
            score = score - self.exposure.penalty(idx)

        order = np.argsort(-score)[: max(top_k * 20, 200)]
        picked = self._diversify(order, idx, top_k, max_per_subcategory)
        rows = idx[picked]
        if self.exposure is not None:
            self.exposure.record(rows)

        out = self.catalog.loc[rows, [
            "parent_id", "product_name", "brand", "category", "sub_category",
            "price_min", "price_max", "price_median", "offer_count",
            "product_url", "image_display_url",
        ]].copy()
        out["match_score"] = np.round(score[picked], 4)
        out["interest_similarity"] = np.round(sim[picked], 4)
        out["occasion_match"] = occ_hit[picked].astype(bool)
        out["budget_fit"] = np.round(pfit[picked], 3)
        return out.reset_index(drop=True), diag

    def rank_subset(self, candidate_ids, age, gender, occasion, budget,
                    interests, top_k=None):
        """Rank a fixed candidate set without diversification or jitter.

        Used for pooled offline evaluation against an annotated reference set.
        Diversification and tie jitter are deliberately disabled: both are
        product behaviours that would distort a pure ranking measurement.
        """
        q = self._embed_query(age, gender, occasion, budget, interests)
        pos = self.catalog.index[self.catalog.parent_id.isin(candidate_ids)].to_numpy()
        if len(pos) == 0:
            return pd.DataFrame()

        sim = self.embeddings[pos] @ q.ravel()
        occ_hit = (self._occ[occasion][pos].astype(float)
                   if occasion in self._occ else np.zeros(len(pos)))
        occ_score = occ_hit * self.weights.get(f"occ_{occasion}", 1.0)

        if gender in ("Female", "Male"):
            g = self._gender[pos]
            gender_score = np.where(g == gender, 1.0, np.where(g == "Unisex", 0.7, 0.2))
        else:
            gender_score = np.full(len(pos), 0.7)

        pfit = price_fit_score(self._price_min[pos], self._price_max[pos], budget)
        score = (self.w_sim * sim + self.w_occasion * occ_score
                 + self.w_gender * gender_score + self.w_price * pfit
                 + self.w_prior * self._prior[pos])

        order = np.argsort(-score)
        out = self.catalog.loc[pos[order], ["parent_id", "product_name"]].copy()
        out["score"] = score[order]
        return out.reset_index(drop=True) if top_k is None else out.head(top_k).reset_index(drop=True)

    def similar_items(self, parent_id, top_k=5):
        """Return the closest catalogue items to a given product."""
        pos = int(self.catalog.index[self.catalog.parent_id == parent_id][0])
        sims = self.embeddings @ self.embeddings[pos]
        order = [i for i in np.argsort(-sims) if i != pos][:top_k]
        out = self.catalog.loc[order, ["parent_id", "product_name", "category",
                                       "price_median"]].copy()
        out["similarity"] = np.round(sims[order], 4)
        return out.reset_index(drop=True)
