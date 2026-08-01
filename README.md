<div align="center">

# 🎁 Giftly — Smart Gift Recommendation System

**Context-aware gift recommendations for the Gulf market**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Graduation Project — Data Science & Machine Learning*

[Live Demo](#running-locally) · [Dataset](#dataset) · [Results](#results) · [API Docs](#api)

</div>

---

## Overview

Giftly turns a recipient description into a ranked shortlist of real, purchasable gifts.
A user provides age, budget, occasion and interests — the system returns the most relevant products from a catalogue of **45,055 real items** scraped from 25 Gulf retailers.

```
Input:  age=25, budget=400 SAR, occasion=Graduation, interests=[Technology]
Output: Ranked list of gifts with images, prices, match scores and purchase links
```

### Why this is not a classic recommender

No user–item interaction history exists in this domain — no purchase logs, no ratings, no repeat visitors. Collaborative filtering is therefore infeasible and is **out of scope by design, not by omission**. The task is formally a **constrained matching and ranking** problem:

| Mechanism | Applies to | Behaviour |
|---|---|---|
| **Hard filter** | Budget, age | Binary. A 900 SAR watch never appears on a 200 SAR budget. |
| **Weighted score** | Occasion, interest, gender, price fit | IDF-weighted similarity over a combined feature space. |
| **Diversity re-rank** | Final top-K | Logarithmic exposure damping prevents the same items dominating every query. |

---

## Results

Evaluated on `gold_test_labelled.csv` — a **held-out, independently annotated** reference set opened once after all tuning was complete.

| Metric | Target | **Result** |
|---|---|---|
| **Precision@5** | ≥ 0.70 | **0.896** ✅ |
| **NDCG@10** | ≥ 0.75 | **0.902** ✅ |
| Precision@1 | — | **0.985** |
| Catalogue coverage @3,000 queries | ≥ 60% | **61.29%** ✅ |
| Slot efficiency | — | **98.5%** |
| Constraint violations (budget & age) | **0%** | **0%** ✅ |
| Empty-result rate | — | **0%** |
| P95 latency | < 300 ms | **40 ms** ✅ |

### Per-occasion breakdown

| Occasion | P@1 | P@5 | NDCG@10 |
|---|---|---|---|
| Eid | 1.000 | 1.000 | 1.000 |
| Birthday | 1.000 | 1.000 | 0.998 |
| Housewarming | 1.000 | 1.000 | 1.000 |
| Anniversary | 1.000 | 1.000 | 0.995 |
| Graduation | 1.000 | 1.000 | 0.987 |
| MothersDay | 1.000 | 0.980 | 0.988 |
| Wedding | 1.000 | 0.800 | 0.839 |
| NewBaby | 1.000 | 0.720 | 0.795 |
| ThankYou | 1.000 | 0.685 | 0.792 |
| FathersDay | 0.850 | 0.770 | 0.622 |
| **OVERALL** | **0.985** | **0.896** | **0.902** |

---

## Dataset

| | |
|---|---|
| Raw rows | 56,280 |
| Source retailers | 25 Gulf stores (Ounass, Jarir, Decathlon, Sun & Sand, Steam, Floward, LEGO…) |
| Deleted (unusable price) | 1,211 (2.15%) |
| Parent products | **45,055** |
| Purchase options retained | **55,069** |
| Categories | 14 (merged from 24) |
| Occasion labels | 10 (multi-label, rule-derived) |
| Interest tags | 15 (multi-label, cross-category) |
| Median price | 379 SAR |

21 of 21 validation checks pass in the cleaning pipeline.
Occasion rules score **macro-F1 = 0.777** against an independently annotated reference set.

Large CSVs are distributed via [GitHub Releases](https://github.com/Mhd366/gift-recommendation-system/releases) — not committed to Git.

```bash
# Download data
gh release download v2.0 --dir data/processed --pattern "catalog_*.csv"
gh release download v2.0 --dir models --pattern "*.joblib"
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Hard Filters (budget + age)            │  ← correctness rules, never relaxed
│  45,055 → candidate pool                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Weighted Scoring                       │
│  ├── Occasion match  (IDF weight 0.32)  │
│  ├── Interest match  (IDF weight 0.30)  │
│  ├── Price fit                  (0.20)  │
│  ├── Category match             (0.13)  │
│  ├── Gender signal              (0.05)  │  ← soft signal, never a filter
│  └── Quality prior              (0.05)  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Diversity Re-ranking                   │
│  ├── Logarithmic exposure damping       │
│  ├── Per-subcategory cap                │
│  └── Query-seeded tie-break jitter      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
            Top-K Results
```

### Feature space

Products occupy **2,203 distinct structural vectors** from categorical features alone (4.9% of 45,055). A **character n-gram text block** over `product_name` raises this to **44,811 distinct vectors (99.5%)**, eliminating the arbitrary within-block ordering that was suppressing the long tail.

Character n-grams were chosen over a sentence transformer because the catalogue is ~30% Arabic — an English-only encoder would discard a third of the inventory. Char n-grams are script-agnostic, fit in 14 seconds, and are reproducible from a seed.

---

## Methodological notes

This project documents its own errors because an evaluation that reports only wins is not an evaluation.

### Four defects found and corrected

| # | Defect | Fix |
|---|---|---|
| 1 | The constraint test checked `price ≤ budget × 1.10` against the same expression — it could never fail | Now tested against the budget the user actually entered |
| 2 | Metrics were computed against `occ_*` columns produced by the same rules the engine ranks on — circular evaluation | Replaced with a pooled evaluation against an independently annotated reference set |
| 3 | Coverage target (60%) was defined without a query count — 600 queries at K=10 can touch at most 13.3% of the catalogue | Restated with its query budget; exposure damping introduced |
| 4 | Three of four proposed rule changes made the system worse | Reported in the notebook rather than discarded silently |

### Train / test separation

Every rule change was tuned on `gold_dev_labelled.csv` (722 rows).
`gold_test_labelled.csv` (300 rows) was opened **once**, after the rules were frozen.

---

## Project structure

```
gift-recommendation-system/
├── app.py                          Streamlit app (deployment entry point)
├── api/
│   └── main.py                     FastAPI service
├── app/
│   └── streamlit_app.py            Streamlit app (local entry point)
├── src/
│   ├── gift_engine.py              Retrieval engine
│   ├── text_block.py               Char n-gram text embeddings
│   ├── exposure.py                 Logarithmic exposure damping
│   └── gift_recommender/
│       ├── preprocessing/          Data cleaning pipeline
│       ├── labeling/               Occasion annotation
│       ├── evaluation/             Ranking metrics
│       └── recommendation/        Legacy engine module
├── notebooks/
│   ├── 01_data_cleaning.ipynb      Raw → clean catalogue (executed, outputs saved)
│   ├── 02_eda.ipynb                18 figures + 12 findings for the modelling team
│   └── 03_modeling_and_evaluation.ipynb   Engine, evaluation, serialisation
├── configs/
│   └── taxonomy.yaml               All cleaning decisions — auditable in Git
├── data/
│   ├── annotations/                Gold label sets (versioned)
│   └── processed/                  Signal weights, EDA findings (versioned)
├── docs/decisions/                 Architectural decision records (ADRs)
├── reports/                        18 EDA figures + evaluation reports
├── models/                         Trained artifact (gitignored, in Releases)
├── requirements.txt
├── runtime.txt
└── railway.json
```

**Two rules about this structure:**
1. Notebooks import from `src/`, never the reverse.
2. `data/raw/` is read-only — every transformation writes to a new stage.

---

## Running locally

```bash
git clone https://github.com/Mhd366/gift-recommendation-system.git
cd gift-recommendation-system

python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Download data and model from the release
gh release download v2.0 --dir data/processed --pattern "catalog_*.csv"
gh release download v2.0 --dir models --pattern "*.joblib"

# Copy engine modules to api/
cp src/gift_engine.py src/text_block.py src/exposure.py api/
```

**Terminal 1 — API:**
```bash
uvicorn api.main:app --port 8000
# Swagger docs at http://localhost:8000/docs
```

**Terminal 2 — UI:**
```bash
streamlit run app/streamlit_app.py
# http://localhost:8501
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/health` | Liveness probe |
| GET | `/options` | Valid occasions, interests, genders |
| GET | `/metrics` | Offline evaluation figures |
| POST | `/recommend` | Ranked recommendations |
| GET | `/product/{id}/options` | Purchase variants |

**Example request:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "Female",
    "occasion": "Graduation",
    "budget": 400,
    "interests": ["Technology"],
    "top_k": 5
  }'
```

---

## Notebooks

| Notebook | Purpose | Key output |
|---|---|---|
| `01_data_cleaning.ipynb` | Raw → clean catalogue | 45,055 products, 21/21 checks pass |
| `02_eda.ipynb` | Exploratory analysis | 18 figures, 12 findings for the ML team |
| `03_modeling_and_evaluation.ipynb` | Engine, tuning, evaluation | Precision@5=0.896, NDCG@10=0.902 |

All notebooks are executed with outputs saved — readable without running.

---

## Known limitations

1. **Pooled evaluation measures ranking, not retrieval.** These figures say the engine orders judged items well. They do not say it finds the right product among 45,055.
2. **Reference labels are machine-generated.** The annotator is independent of the rule engine, but it is not human ground truth. See `docs/decisions/0007`.
3. **FathersDay is the weakest occasion** (NDCG@10 = 0.622). Root cause: only 12% of the catalogue is labelled `Male`, and that label was built by keyword matching. It is a data defect, not a ranking defect.
4. **Gender is a soft signal (weight 0.05), never a hard filter.** Filtering on gender encodes stereotypes and shrinks the candidate pool for no measured relevance gain.
5. **Block weights are expert-set**, not learned. Unlearnable without interaction data. A/B test once traffic exists.

---

## Phase 2 — when interaction data exists

Once the API has logged purchases or clicks, the correct next model is a **learning-to-rank** objective (LambdaMART or XGBoost `rank:pairwise`) trained on those outcomes, using the current cosine score, budget fit and cluster id as features. The current engine becomes the candidate generator and the supervised ranker re-orders its output.

Supervised learning was not excluded because those algorithms are weak — it was excluded because on this dataset, at this phase, there is nothing for them to learn from.

---

## Legal & ethical notes

- Product data was collected in accordance with each site's `robots.txt` and terms of service.
- Images are **hotlinked to the original retailer**, not redistributed.
- The system links to retailers; it processes no payments and stores no personal data.
- **Gender is a soft signal, never a hard filter.** See above.

---

## License

MIT — see [LICENSE](LICENSE).

Product names, descriptions and images belong to their respective retailers and are referenced for academic, non-commercial purposes.

---

<div align="center">

Built with 🎁 for the Gulf market

*Graduation Project — 2026*

</div>
