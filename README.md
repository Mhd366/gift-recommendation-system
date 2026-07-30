# Smart Gift Recommendation System

Context-aware gift recommender for the Gulf market. The user describes the
recipient — age, budget, occasion, interests — and the system returns a ranked
shortlist of real, purchasable products.

## Team ownership

| Area | Folders | Owner |
|---|---|---|
| Data & EDA | `data/`, `notebooks/01`, `notebooks/02`, `reports/`, `preprocessing/`, `labeling/` | Data |
| Modelling | `models/`, `notebooks/03`, `recommendation/`, `evaluation/` | ML |
| API & frontend | `api/`, `app/` | API |

Work on your own branch and open a pull request into `dev`. `main` is protected.

## Why not collaborative filtering

No user–item interaction history exists in this domain, so the task is
**constrained matching and ranking**, not classic recommendation.

| Mechanism | Applies to | Behaviour |
|---|---|---|
| Hard filter | budget, age | Binary, never relaxed |
| Weighted score | occasion, interest, category, price fit | Inverse-frequency weighted |
| Re-rank | diversity | MMR with a per-category cap |

## Dataset

| | |
|---|---|
| Raw rows | 56,280 from 25 Gulf retailers |
| Deleted (unusable price) | 1,211 (2.15%) |
| Parent products | 45,055 |
| Purchase options retained | 55,069 |
| Categories | 14, merged from 24 |
| Occasion labels | 10, multi-label |
| Interest tags | 15, multi-label, cross-category |

21 of 21 validation checks pass. Occasion rules score macro-F1 0.777 against an
independently annotated reference set.

## Handoff contract

| File | Consumer | Notes |
|---|---|---|
| `catalog_clean.csv` | ML, API | One row per product |
| `catalog_offers.csv` | API | Purchase options, joined on `parent_id` |
| `signal_weights.csv` | ML | Inverse-frequency ranking weights |
| `eda_findings.csv` | ML | **Read before modelling** |

Large CSVs are distributed via GitHub Releases.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
cp .env.example .env
gh release download v1.0 --dir data/processed --pattern "catalog_*.csv"
```

## Critical findings for modelling

- Two occasion labels cover almost the entire catalogue and carry no ranking
  signal alone; every occasion match must be multiplied by its weight.
- `description` is template-generated from the structured fields and adds no
  independent semantic content; use `product_name` for text features.
- `source_site` correlates with category and price and must be excluded from any
  model intended to generalise.
- Price is heavily right-skewed; use `price_band` or log price, never raw price
  in a distance metric.
- 15 of 70 category-budget cells hold fewer than 50 products; these are supply
  gaps, not model failures.

## Known limitations

- Occasion labels are rule-derived and validated against a machine-annotated
  reference set, not human ground truth. See `docs/decisions/0007`.
- Sub-category is unverifiable for a share of products; flagged via
  `subcategory_confidence` and used to demote, never to delete.
- Gender is a low-weight soft signal, never a hard filter.

## Commits

Conventional Commits with a team scope:
## License

MIT. Product data belongs to the respective retailers and is referenced for
academic, non-commercial use.
