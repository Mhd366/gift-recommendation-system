"""Score the rule-based occasion engine against an annotated gold set.

Reports per-label precision, recall and F1, plus macro and micro averages and
Hamming loss, and writes a Markdown report.

Usage::

    python -m gift_recommender.evaluation.evaluate_labels \
        --gold data/annotations/gold_test_labelled.csv \
        --catalog data/processed/catalog_clean.csv \
        --report reports/occasion_rule_evaluation.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

logger = logging.getLogger(__name__)

# Gold label -> the corresponding rule-engine column in the clean catalogue.
LABEL_TO_RULE: dict[str, str] = {
    "Birthday": "occ_birthday",
    "Graduation": "occ_graduation",
    "Anniversary": "occ_anniversary",
    "Wedding": "occ_wedding",
    "Eid": "occ_eid__religious",
    "MothersDay": "occ_mothers_day",
    "FathersDay": "occ_fathers_day",
    "NewBaby": "occ_new_baby",
    "Housewarming": "occ_housewarming",
    "ThankYou": "occ_thank_you",
}


def evaluate(gold: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Compute per-label classification metrics for the rule engine.

    Args:
        gold: Annotated sample with one binary column per occasion.
        catalog: Clean catalogue carrying the rule-engine ``occ_*`` columns.

    Returns:
        A frame indexed by label with support, prevalence and P/R/F1.
    """
    merged = gold.merge(
        catalog[["Product ID", *LABEL_TO_RULE.values()]], on="Product ID", how="inner"
    )
    logger.info("matched %d of %d gold rows against the catalogue", len(merged), len(gold))

    rows = []
    for label, rule_col in LABEL_TO_RULE.items():
        y_true = merged[label].astype(int).to_numpy()
        y_pred = merged[rule_col].astype(bool).astype(int).to_numpy()
        rows.append({
            "label": label,
            "support": int(y_true.sum()),
            "gold_rate": y_true.mean(),
            "rule_rate": y_pred.mean(),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows).set_index("label")


def summarise(gold: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, float]:
    """Compute macro/micro F1 and Hamming loss across all labels."""
    merged = gold.merge(
        catalog[["Product ID", *LABEL_TO_RULE.values()]], on="Product ID", how="inner"
    )
    y_true = merged[list(LABEL_TO_RULE)].astype(int).to_numpy()
    y_pred = merged[list(LABEL_TO_RULE.values())].astype(bool).astype(int).to_numpy()
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
    }


def to_markdown(table: pd.DataFrame, summary: dict[str, float], n: int) -> str:
    """Render the evaluation as a Markdown report."""
    lines = [
        "# Occasion rule engine — evaluation",
        "",
        f"Gold set: **{n} products**, stratified over Category x price band.",
        "",
        "> **Label provenance: machine annotator (archetype-based), not human review.**",
        "> These figures measure agreement between two independent labelling methods,",
        "> not accuracy against human judgement. See docs/decisions/0007.",
        "",
        "## Per-label results",
        "",
        "| Label | Support | Gold rate | Rule rate | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in table.iterrows():
        lines.append(
            f"| {label} | {int(r['support'])} | {r['gold_rate']:.1%} | {r['rule_rate']:.1%} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Macro-F1 | {summary['macro_f1']:.3f} |",
        f"| Micro-F1 | {summary['micro_f1']:.3f} |",
        f"| Hamming loss | {summary['hamming_loss']:.3f} |",
        f"| Exact match | {summary['exact_match']:.1%} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate occasion rules against gold.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gold = pd.read_csv(args.gold, encoding="utf-8-sig")
    catalog = pd.read_csv(args.catalog, encoding="utf-8-sig")

    table = evaluate(gold, catalog)
    summary = summarise(gold, catalog)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(to_markdown(table, summary, len(gold)), encoding="utf-8")

    logger.info("\n%s", table.round(3).to_string())
    logger.info("\nmacro-F1 %.3f | micro-F1 %.3f | hamming %.3f | exact %.1f%%",
                summary["macro_f1"], summary["micro_f1"],
                summary["hamming_loss"], summary["exact_match"] * 100)
    logger.info("\nwrote %s", args.report)


if __name__ == "__main__":
    main()
