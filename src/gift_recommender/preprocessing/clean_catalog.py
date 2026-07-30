"""Clean the raw gift catalogue into a single analysis-ready dataset.

Reads one raw CSV and writes one clean CSV. Every transformation is driven by
``configs/taxonomy.yaml`` so that judgement calls stay auditable in version
control rather than buried in code.

Pipeline stages, in order::

    load -> drop_redundant -> normalise_categories -> normalise_age
         -> apply_safety_floors -> clean_prices -> deduplicate
         -> derive_occasions -> derive_flags -> finalise

Usage::

    python -m gift_recommender.preprocessing.clean_catalog \
        --input data/raw/catalog_raw.csv \
        --output data/processed/catalog_clean.csv \
        --taxonomy configs/taxonomy.yaml \
        --report reports/cleaning_report.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Columns removed outright. Each entry carries the reason it is dropped, so a
# reviewer never has to guess. See docs/decisions/0002-redundant-columns.md
REDUNDANT_COLUMNS: dict[str, str] = {
    "Currency": "single constant value (SAR)",
    "Interest Category": "byte-identical to Category in 100% of rows",
    "Recommendation Tags": "string concatenation of six other columns",
    "Data Quality": "metadata describing nulls, not a product attribute",
    "Recipient Type": "conflates gender, age and household axes (ADR 0005)",
    "Price Tier": "two incompatible binning schemes; recomputed from price",
    "Luxury Level": "derived from price; recomputed deterministically",
    "Source File": "scrape provenance, not a product attribute",
}

# The canonical price bands, recomputed from the numeric price so that a single
# consistent scheme replaces the two conflicting ones in the raw data.
PRICE_BANDS: list[tuple[float, float, str]] = [
    (0.0, 100.0, "Under 100 SAR"),
    (100.0, 300.0, "100-299 SAR"),
    (300.0, 700.0, "300-699 SAR"),
    (700.0, 1500.0, "700-1499 SAR"),
    (1500.0, float("inf"), "1500+ SAR"),
]


@dataclass
class CleaningReport:
    """Accumulates row-level accounting for every stage of the pipeline."""

    rows_in: int = 0
    rows_out: int = 0
    stages: list[tuple[str, int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, stage: str, before: int, after: int) -> None:
        """Log one stage's row delta."""
        self.stages.append((stage, before, after))
        delta = before - after
        if delta:
            logger.info("%-24s %6d -> %6d  (removed %d)", stage, before, after, delta)
        else:
            logger.info("%-24s %6d rows unchanged", stage, before)

    def note(self, text: str) -> None:
        """Attach a free-text observation to the report."""
        self.notes.append(text)
        logger.info("  note: %s", text)

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines = ["# Cleaning report", "", "| Stage | Rows before | Rows after | Removed |",
                 "|---|---:|---:|---:|"]
        for stage, before, after in self.stages:
            lines.append(f"| {stage} | {before:,} | {after:,} | {before - after:,} |")
        retained = self.rows_out / self.rows_in * 100 if self.rows_in else 0.0
        lines += ["", f"**{self.rows_in:,} in -> {self.rows_out:,} out "
                      f"({retained:.1f}% retained)**", "", "## Notes", ""]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def load_taxonomy(path: Path) -> dict[str, Any]:
    """Load the taxonomy configuration file."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_raw(path: Path, report: CleaningReport) -> pd.DataFrame:
    """Read the raw CSV, stripping the UTF-8 BOM emitted by the scraper."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    report.rows_in = len(df)
    logger.info("loaded %d rows x %d columns", len(df), df.shape[1])
    return df


def drop_redundant(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove columns that duplicate, derive from, or describe other columns."""
    present = [c for c in REDUNDANT_COLUMNS if c in df.columns]
    for col in present:
        report.note(f"dropped column `{col}` - {REDUNDANT_COLUMNS[col]}")
    return df.drop(columns=present)


def normalise_categories(df: pd.DataFrame, tax: dict[str, Any],
                         report: CleaningReport) -> pd.DataFrame:
    """Collapse near-duplicate category and gift-type labels onto canonics."""
    before_cat = df["Category"].nunique()
    df["Category"] = df["Category"].map(tax["category_map"]).fillna(df["Category"])
    report.note(f"Category vocabulary {before_cat} -> {df['Category'].nunique()} values")

    if "Gift Type" in df.columns:
        before_gt = df["Gift Type"].nunique()
        df["Gift Type"] = df["Gift Type"].map(tax["gift_type_map"]).fillna(df["Gift Type"])
        report.note(f"Gift Type vocabulary {before_gt} -> {df['Gift Type'].nunique()} values")

    df["Sub Category"] = df["Sub Category"].astype(str).str.strip().str.title()
    return df


def normalise_age(df: pd.DataFrame, tax: dict[str, Any],
                  report: CleaningReport) -> pd.DataFrame:
    """Convert the textual age bucket into numeric ``min_age`` / ``max_age``.

    String buckets cannot be compared against a user's stated age. Numeric
    bounds turn age from a fuzzy label into an enforceable hard constraint.
    """
    ranges: dict[str, dict[str, int]] = tax["age_ranges"]
    df["min_age"] = df["Age Group"].map(lambda v: ranges.get(v, {}).get("min", 0))
    df["max_age"] = df["Age Group"].map(lambda v: ranges.get(v, {}).get("max", 99))

    unmapped = df.loc[~df["Age Group"].isin(ranges), "Age Group"].unique()
    if len(unmapped):
        report.note(f"unmapped age labels defaulted to 0-99: {list(unmapped)}")
    return df


def apply_safety_floors(df: pd.DataFrame, tax: dict[str, Any],
                        report: CleaningReport) -> pd.DataFrame:
    """Raise ``min_age`` for sources whose content carries no age rating.

    Steam titles ship without PEGI/ESRB data in this catalogue and include
    violent and adult-themed games. Rather than risk surfacing them to a child
    query under filter relaxation, the whole source is floored at 13+.
    """
    floors: dict[str, int] = tax.get("safety_floors", {}).get("by_source_site", {})
    for site, floor in floors.items():
        mask = df["Source Site"] == site
        if not mask.any():
            continue
        df.loc[mask, "min_age"] = df.loc[mask, "min_age"].clip(lower=floor)
        report.note(f"safety floor min_age>={floor} applied to {int(mask.sum()):,} "
                    f"rows from `{site}` (no content rating available)")
    df["age_locked"] = df["Source Site"].isin(floors)
    return df


def clean_prices(df: pd.DataFrame, cfg: dict[str, Any],
                 report: CleaningReport) -> pd.DataFrame:
    """Drop unusable prices and recompute a single consistent band scheme.

    Price is a hard constraint, so a missing value cannot be imputed: an
    imputed price could place an item inside a budget it does not belong in.
    """
    col = "Price (SAR)"
    df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df)
    missing = int(df[col].isna().sum())
    df = df[df[col].notna()]
    report.record("drop missing price", n, len(df))
    if missing:
        report.note(f"{missing:,} rows had no price; price is a hard constraint "
                    "and cannot be imputed")

    n = len(df)
    zero = int((df[col] <= 0).sum())
    df = df[df[col] > 0]
    report.record("drop free items", n, len(df))
    if zero:
        report.note(f"{zero:,} zero-price items removed (free-to-play Steam titles); "
                    "a free item is not a gift")

    n = len(df)
    ceiling = float(cfg["max_valid"])
    above = int((df[col] > ceiling).sum())
    df = df[df[col] <= ceiling]
    report.record("drop price outliers", n, len(df))
    if above:
        report.note(f"{above:,} items above {ceiling:,.0f} SAR removed "
                    "(outside any realistic gift budget)")

    df["price_band"] = pd.cut(
        df[col],
        bins=[b[0] for b in PRICE_BANDS] + [PRICE_BANDS[-1][1]],
        labels=[b[2] for b in PRICE_BANDS],
        right=False,
    )
    report.note("price_band recomputed from numeric price using one consistent scheme")
    return df


def deduplicate(df: pd.DataFrame, keys: list[str], report: CleaningReport) -> pd.DataFrame:
    """Collapse colour/size variants of the same product to one representative.

    Variants are near-identical items that would otherwise flood a result list
    with five versions of the same shoe, destroying intra-list diversity. The
    cheapest variant is kept because it maximises budget compatibility.
    """
    n = len(df)
    keys = [k for k in keys if k in df.columns]
    df = (df.sort_values("Price (SAR)")
            .drop_duplicates(subset=keys, keep="first")
            .sort_index())
    report.record("deduplicate variants", n, len(df))
    report.note(f"variants collapsed on {keys}, keeping the cheapest representative")
    return df


def _matches_occasion(row: pd.Series, rule: dict[str, Any]) -> bool:
    """Return True when a product satisfies every condition of one rule."""
    cats = rule.get("include_categories", [])
    if cats != ["*"] and row["Category"] not in cats:
        return False
    if row["Category"] in rule.get("exclude_categories", []):
        return False
    if row["Gift Type"] in rule.get("exclude_gift_types", []):
        return False
    if "min_age" in rule and row["max_age"] < rule["min_age"]:
        return False
    if "max_age" in rule and row["min_age"] > rule["max_age"]:
        return False
    genders = rule.get("gender_target")
    if genders and row["Gender Target"] not in genders:
        return False
    price = row["Price (SAR)"]
    if "min_price" in rule and price < rule["min_price"]:
        return False
    if "max_price" in rule and price > rule["max_price"]:
        return False
    return True


def derive_occasions(df: pd.DataFrame, tax: dict[str, Any],
                     report: CleaningReport) -> pd.DataFrame:
    """Re-derive multi-label occasions using inclusion rather than selection.

    An occasion is a social context, not a product category. The raw catalogue
    tagged Eid on 400 of 56,280 rows because it matched on category name; in a
    Gulf market Eid is arguably the single largest gifting occasion. These
    rules ask "would this be acceptable to give at X?" instead of "is this an
    X product?".

    This is a transparent heuristic baseline, not a learned model. Its accuracy
    is measured against the human-labelled gold set in ``data/annotations/``.
    """
    rules: dict[str, dict[str, Any]] = tax["occasion_rules"]
    for name, rule in rules.items():
        col = f"occ_{name.lower().replace(' ', '_').replace('/', '').replace(chr(39), '')}"
        df[col] = df.apply(lambda r, rl=rule: _matches_occasion(r, rl), axis=1)
        report.note(f"occasion `{name}`: {int(df[col].sum()):,} products "
                    f"({df[col].mean() * 100:.1f}%)")

    occ_cols = [c for c in df.columns if c.startswith("occ_")]
    df["Occasions"] = df[occ_cols].apply(
        lambda r: " | ".join(
            n for n, c in zip(rules.keys(), occ_cols, strict=True) if r[c]
        ) or "Any Occasion",
        axis=1,
    )
    df["occasion_count"] = df[occ_cols].sum(axis=1)
    report.note("`General / Any Occasion` dropped as a label - it appeared on 96.7% "
                "of raw rows and carried no discriminative power")
    return df


def derive_flags(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Add quality and presentation flags used downstream by the ranker."""
    df["Brand"] = df["Brand"].fillna("Unbranded")
    df["has_image"] = df["Image URL"].notna()
    missing_img = int((~df["has_image"]).sum())
    report.note(f"{missing_img:,} products lack an image; flagged rather than dropped "
                "so the ranker can demote instead of losing catalogue coverage")

    df["is_family_gift"] = df["Category"].isin(["Home & Living", "Food & Sweets"])
    df["Description"] = df["Description"].astype(str).str.strip()
    return df


def finalise(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Order columns for readability and reset the index."""
    order = [
        "Product ID", "Product Name", "Brand", "Price (SAR)", "price_band",
        "Category", "Sub Category", "Gift Type",
        "Gender Target", "Age Group", "min_age", "max_age", "age_locked",
        "Occasions", "occasion_count",
        "Description", "Product URL", "Image URL", "has_image",
        "Source Site", "Source Category", "is_family_gift",
    ]
    order += [c for c in df.columns if c.startswith("occ_")]
    df = df[[c for c in order if c in df.columns]].reset_index(drop=True)
    report.rows_out = len(df)
    return df


def clean(raw_path: Path, taxonomy_path: Path) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return the result with its report."""
    tax = load_taxonomy(taxonomy_path)
    report = CleaningReport()

    df = load_raw(raw_path, report)
    df = drop_redundant(df, report)
    df = normalise_categories(df, tax, report)
    df = normalise_age(df, tax, report)
    df = apply_safety_floors(df, tax, report)
    df = clean_prices(df, {"max_valid": 50_000}, report)
    df = deduplicate(df, ["Product Name", "Brand", "Price (SAR)", "Source Site"], report)
    df = derive_occasions(df, tax, report)
    df = derive_flags(df, report)
    return finalise(df, report), report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Clean the raw gift catalogue.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/taxonomy.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/cleaning_report.md"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df, report = clean(args.input, args.taxonomy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.to_markdown(), encoding="utf-8")

    logger.info("\nwrote %s (%d rows x %d cols)", args.output, len(df), df.shape[1])
    logger.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
