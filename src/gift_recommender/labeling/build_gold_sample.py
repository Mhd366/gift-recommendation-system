"""Draw a stratified sample from the clean catalogue for human annotation.

The sample must represent the catalogue, not merely be random: a uniform draw
over 48,924 rows would return ~19% Gaming and almost no Flowers, so rare
categories would carry no measurable accuracy at all.

Stratification is over ``Category x price_band``. Within each stratum the draw
is proportional to stratum size, with a floor so that thin strata still receive
at least a few rows.

The output is split into two disjoint files:

``gold_test.csv``
    Held out. Annotated by a human with no machine assistance. This is the only
    file used to report final accuracy.
``gold_dev.csv``
    Pre-filled with suggestions for human review. Used to tune rules and
    thresholds. Never used to report accuracy.

Usage::

    python -m gift_recommender.labeling.build_gold_sample \
        --input data/processed/catalog_clean.csv \
        --outdir data/annotations \
        --n-test 300 --n-dev 700 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The label space a human fills in. One binary column per occasion, so that the
# annotator answers a series of yes/no questions rather than composing a set.
OCCASION_LABELS: list[str] = [
    "Birthday", "Graduation", "Anniversary", "Wedding", "Eid",
    "MothersDay", "FathersDay", "NewBaby", "Housewarming", "ThankYou",
]

# Columns shown to the annotator. Deliberately excludes every rule-derived
# field (occ_*, Occasions, occasion_count) so the human is not anchored by the
# very predictions we intend to score.
CONTEXT_COLUMNS: list[str] = [
    "Product ID", "Product Name", "Brand", "Price (SAR)",
    "Category", "Sub Category", "Gift Type", "Product URL",
]

# Extra judgement columns: the annotator also audits the two fields whose
# provenance we distrust (both were built by keyword matching on product name).
AUDIT_COLUMNS: dict[str, str] = {
    "audit_gender": "Correct Gender Target? Female / Male / Unisex",
    "audit_min_age": "Youngest sensible recipient age (integer)",
    "notes": "Free text: ambiguity, wrong category, dead link, anything odd",
}


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    strata: list[str],
    seed: int,
    floor: int = 2,
) -> pd.DataFrame:
    """Draw ``n`` rows proportionally across ``strata``, with a per-stratum floor.

    Args:
        df: Source frame.
        n: Target sample size. The result may exceed this slightly when the
            floor forces extra draws from thin strata.
        strata: Columns whose cross-product defines the strata.
        seed: Random seed for reproducibility.
        floor: Minimum rows drawn from any non-empty stratum.

    Returns:
        The sampled rows.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby(strata, observed=True)
    total = len(df)

    parts: list[pd.DataFrame] = []
    for _, group in groups:
        quota = max(floor, round(n * len(group) / total))
        quota = min(quota, len(group))
        idx = rng.choice(group.index.to_numpy(), size=quota, replace=False)
        parts.append(df.loc[idx])

    sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
    logger.info("drew %d rows across %d strata", len(sample), groups.ngroups)
    return sample


def build_workbook(sample: pd.DataFrame, blank: bool) -> pd.DataFrame:
    """Shape a sample into an annotation workbook.

    Args:
        sample: Rows drawn from the catalogue.
        blank: When True every label column is left empty (the held-out set).
            When False the columns are present but still empty, ready to be
            populated with reviewable suggestions.

    Returns:
        A frame with context columns followed by empty label columns.
    """
    book = sample[CONTEXT_COLUMNS].copy()
    for label in OCCASION_LABELS:
        book[label] = "" if blank else pd.NA
    for col in AUDIT_COLUMNS:
        book[col] = ""
    return book.reset_index(drop=True)


def write_guidelines(path: Path) -> None:
    """Write the annotation protocol that accompanies the workbooks."""
    path.write_text(
        """# Annotation guidelines - occasion labelling

## The question you are answering

For each product, for each occasion, answer one question:

> **Would giving this item at this occasion be socially acceptable?**

Not "is this the ideal gift". Not "is this an X-themed product". Acceptable.

Mark `1` for yes, `0` for no. Never leave a cell empty.

## Rules

1. **Multi-label.** A watch can be `1` for Birthday, Graduation and Anniversary
   at once. Most products will have several.
2. **Judge the product, not the shop.** Ignore which retailer it came from.
3. **Judge from the name, category and price.** Open the URL only when the name
   is genuinely opaque.
4. **Price matters.** A 40 SAR keychain is not a Wedding gift. A 12,000 SAR
   bracelet is not a Thank You gift.
5. **Cultural context is Saudi/Gulf.** Eid is a major gifting occasion across
   all ages. Valentine's is not observed in the same way.
6. **When you cannot decide, mark `0`** and write why in `notes`. A forced
   guess pollutes the ground truth; a note tells us the rule needs work.

## Occasion definitions

| Label | Give it to | Typical range |
|---|---|---|
| Birthday | Anyone | Any price |
| Graduation | Student finishing a stage, 16+ | Mid to high |
| Anniversary | Romantic partner, 18+ | Mid to high |
| Wedding | Couple or household | Mid to high, often household goods |
| Eid | Anyone, all ages, family-wide | Any price; sweets, perfume, clothing, toys |
| MothersDay | Mother figure | Any price |
| FathersDay | Father figure | Any price |
| NewBaby | Usually the **parents**, sometimes the infant | Low to mid |
| Housewarming | New home owner or household | Low to mid |
| ThankYou | Colleague, host, helper | Low to mid, never intimate |

## Audit columns

- `audit_gender` - is the stated Gender Target right? Both `Gender Target` and
  `Age Group` in the source catalogue were built by keyword matching on the
  product name and their accuracy is unverified. This column measures it.
- `audit_min_age` - the youngest age at which this gift makes sense.
- `notes` - anything wrong: wrong category, dead link, duplicate, unsafe item.

## Rules for the held-out set (`gold_test.csv`)

- Label it **without** looking at `gold_dev.csv`, the pipeline output, or
  `configs/taxonomy.yaml`.
- Label it **once**. Do not revise it after seeing model results. Revising a
  test set to match your model is how benchmarks become meaningless.
- If a definition above turns out to be unclear, fix the definition, then
  relabel from scratch - do not patch individual rows.
""",
        encoding="utf-8",
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build gold annotation samples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-dev", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    logger.info("catalogue: %d rows", len(df))

    pool = stratified_sample(df, args.n_test + args.n_dev,
                             ["Category", "price_band"], args.seed)

    test = pool.iloc[: args.n_test]
    dev = pool.iloc[args.n_test :]

    build_workbook(test, blank=True).to_csv(
        args.outdir / "gold_test.csv", index=False, encoding="utf-8-sig")
    build_workbook(dev, blank=False).to_csv(
        args.outdir / "gold_dev.csv", index=False, encoding="utf-8-sig")
    write_guidelines(args.outdir / "GUIDELINES.md")

    logger.info("wrote gold_test.csv  (%d rows) - HUMAN ONLY, unassisted", len(test))
    logger.info("wrote gold_dev.csv   (%d rows) - pre-fill then review", len(dev))
    logger.info("wrote GUIDELINES.md")


if __name__ == "__main__":
    main()
