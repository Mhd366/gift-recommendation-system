"""Archetype-based occasion annotator for the gold sets.

Improves on ``suggest_labels`` by reasoning over the *sub-category* vocabulary
(70 values) rather than product-name keywords alone. Sub-category identifies
what an item **is**; the name lexicon only fires when a recognisable term
appears, which left 50% of rows without any signal.

Independence from the rule engine
---------------------------------
``configs/taxonomy.yaml`` derives occasions from ``Category`` (14 values),
gift type, age and gender. This module never reads ``Category``. It reads
``Sub Category``, product name and price. Agreement between the two is
therefore informative rather than tautological.

Method
------
1. Map each sub-category to a **product archetype** (a social gift class).
2. Give each archetype a base occasion profile.
3. Apply **price gates**: occasions carry social price expectations that
   override archetype. A 4 SAR highlighter is not a wedding gift; a 28,000 SAR
   handbag is not a thank-you gift.
4. Apply **name overrides** for signals the archetype cannot capture
   (explicitly gendered, baby-specific, or religious items).

Usage::

    python -m gift_recommender.labeling.annotate_v2 \
        --input data/annotations/gold_test.csv \
        --output data/annotations/gold_test_labelled.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LABELS: list[str] = [
    "Birthday", "Graduation", "Anniversary", "Wedding", "Eid",
    "MothersDay", "FathersDay", "NewBaby", "Housewarming", "ThankYou",
]

# --------------------------------------------------------------------------
# Sub-category -> archetype. Archetypes are social gift classes, not retail
# taxonomies: what the item MEANS when handed to someone.
# --------------------------------------------------------------------------
ARCHETYPE: dict[str, str] = {
    "Jewellery": "adornment", "Jelewry": "adornment", "Watches": "milestone",
    "Perfume & Fragrance": "personal_luxury", "Skincare": "personal_care",
    "Makeup": "personal_care", "Haircare": "personal_care", "Body Care": "personal_care",
    "Pc Games": "entertainment", "Gaming": "entertainment",
    "Gaming Hardware & Accessories": "tech", "Electronics": "tech",
    "Toys": "child_play", "Educational Toys": "child_learning",
    "Kids Gifts": "child_play", "Baby Products": "infant",
    "Shoes": "apparel", "Clothing": "apparel", "Accessories": "apparel",
    "Bags & Wallets": "apparel_luxury", "Fashion": "apparel",
    "Fashion & Accessories": "apparel",
    "Chocolate": "edible", "Cakes & Desserts": "edible", "Coffee & Tea": "edible",
    "Dates": "edible_cultural",
    "Flowers": "floral", "Plants": "floral",
    "Dinning": "household", "Kitchen & Dining": "household", "Bedroom": "household",
    "Living Room": "household", "Home Décor": "household", "Home Gifts": "household",
    "Candles & Home Fragrance": "household_soft", "Gifts For Home": "household",
    "Office": "professional", "Books": "intellectual", "Books & Learning": "intellectual",
    "Arts & Crafts": "creative",
    "Men'S Sports": "sport", "Football": "sport", "Swimming": "sport",
    "Outdoors": "sport", "Boxing": "sport", "Cardio": "sport", "Padel": "sport",
    "Archery": "sport", "Yoga": "sport", "Cycling": "sport", "Basketball": "sport",
    "Sports & Fitness": "sport",
    "Gift Sets": "curated", "Best Seller Gifts": "curated", "Gifts": "curated",
    "Gift Cards & Vouchers": "voucher",
    "Religious Gifts": "religious",
    "Birthday Gift": "occasion_birthday", "Graduation Gift": "occasion_graduation",
    "For Him Gift": "masculine", "Gifts For Him": "masculine",
    "Gifts For Men": "masculine", "Mens Gift": "masculine",
    "Gift For Her": "feminine", "Gifts For Her": "feminine",
    "Gifts For Women": "feminine", "Womens Gift": "feminine",
}

# Archetype -> occasions it is socially acceptable for, before price gating.
PROFILE: dict[str, set[str]] = {
    "adornment":       {"Birthday", "Graduation", "Anniversary", "Wedding", "Eid", "MothersDay"},
    "milestone":       {"Birthday", "Graduation", "Anniversary", "Eid", "FathersDay"},
    "personal_luxury": {"Birthday", "Anniversary", "Eid", "MothersDay", "FathersDay", "ThankYou"},
    "personal_care":   {"Birthday", "Eid", "MothersDay", "ThankYou"},
    "entertainment":   {"Birthday", "Graduation", "Eid"},
    "tech":            {"Birthday", "Graduation", "Eid", "FathersDay"},
    "child_play":      {"Birthday", "Eid", "NewBaby"},
    "child_learning":  {"Birthday", "Graduation", "Eid", "NewBaby"},
    "infant":          {"NewBaby", "Birthday", "Eid"},
    "apparel":         {"Birthday", "Eid", "Graduation"},
    "apparel_luxury":  {"Birthday", "Anniversary", "Graduation", "Eid", "MothersDay"},
    "edible":          {"Birthday", "Eid", "ThankYou", "MothersDay", "Housewarming", "NewBaby"},
    "edible_cultural": {"Eid", "ThankYou", "Housewarming"},
    "floral":          {"Birthday", "Anniversary", "MothersDay", "ThankYou",
                        "Housewarming", "NewBaby", "Wedding"},
    "household":       {"Wedding", "Housewarming", "Birthday", "Eid"},
    "household_soft":  {"Housewarming", "ThankYou", "Birthday", "MothersDay", "Wedding"},
    "professional":    {"Graduation", "Birthday", "ThankYou", "FathersDay"},
    "intellectual":    {"Birthday", "Graduation", "Eid", "ThankYou"},
    "creative":        {"Birthday", "Eid", "ThankYou", "Graduation"},
    "sport":           {"Birthday", "Graduation", "Eid", "FathersDay"},
    "curated":         {"Birthday", "Eid", "ThankYou", "MothersDay", "Housewarming"},
    "voucher":         {"Birthday", "Graduation", "Eid", "ThankYou"},
    "religious":       {"Eid", "Wedding", "ThankYou"},
    "masculine":       {"Birthday", "Graduation", "Eid", "FathersDay", "Anniversary"},
    "feminine":        {"Birthday", "Anniversary", "Eid", "MothersDay"},
    "occasion_birthday":   {"Birthday", "Eid"},
    "occasion_graduation": {"Graduation", "Birthday"},
    "unknown":         {"Birthday", "Eid"},
}

# Socially expected price windows, in SAR. An item outside the window is
# removed from that occasion regardless of archetype.
PRICE_GATE: dict[str, tuple[float, float]] = {
    "Birthday":     (10, 100_000),
    "Graduation":   (80, 100_000),
    "Anniversary":  (150, 100_000),
    "Wedding":      (200, 100_000),
    "Eid":          (10, 100_000),
    "MothersDay":   (50, 100_000),
    "FathersDay":   (50, 100_000),
    "NewBaby":      (30, 3_000),
    "Housewarming": (40, 3_000),
    "ThankYou":     (20, 600),
}

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")

NAME_RULES: list[tuple[list[str], set[str], set[str]]] = [
    # (terms, force-on, force-off)
    (["baby", "newborn", "infant", "رضيع", "مولود", "حديثي الولاده"],
     {"NewBaby"}, {"Anniversary", "Wedding", "Graduation", "FathersDay"}),
    (["ramadan", "eid", "رمضان", "عيد", "مصحف", "سجاده"],
     {"Eid"}, set()),
    (["wedding", "bridal", "زفاف", "عروس"],
     {"Wedding"}, {"NewBaby", "Graduation"}),
    (["graduation", "تخرج"],
     {"Graduation"}, set()),
    (["lingerie", "underwear", "ملابس داخليه"],
     set(), {"ThankYou", "Graduation", "MothersDay", "FathersDay",
             "NewBaby", "Housewarming", "Wedding"}),
    (["lego", "pokemon", "disney", "barbie", "بوكيمون", "ديزني", "اطفال", "للاطفال"],
     set(), {"Anniversary", "Wedding", "MothersDay", "FathersDay", "Housewarming"}),
]


def normalise(text: str) -> str:
    """Lowercase, strip Arabic diacritics, and unify letter variants."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = ARABIC_DIACRITICS.sub("", text)
    for src, dst in (("أإآ", "ا"), ("ى", "ي"), ("ة", "ه")):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def annotate(name: str, sub_category: str, price: float) -> dict[str, int]:
    """Assign binary occasion labels to one product.

    Args:
        name: Product name, Arabic or English.
        sub_category: Retail sub-category, used to resolve the archetype.
        price: Price in SAR.

    Returns:
        Mapping from occasion label to 0 or 1.
    """
    archetype = ARCHETYPE.get(str(sub_category).strip(), "unknown")
    active = set(PROFILE[archetype])

    norm = normalise(name)
    for terms, force_on, force_off in NAME_RULES:
        if any(normalise(t) in norm for t in terms):
            active |= force_on
            active -= force_off

    if pd.notna(price):
        for label in list(active):
            low, high = PRICE_GATE[label]
            if not low <= float(price) <= high:
                active.discard(label)

    return {label: int(label in active) for label in LABELS}


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Archetype-based occasion annotator.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    labels = pd.DataFrame(
        [annotate(r["Product Name"], r["Sub Category"], r["Price (SAR)"])
         for _, r in df.iterrows()],
        index=df.index,
    )
    for col in LABELS:
        df[col] = labels[col]

    df["archetype"] = df["Sub Category"].map(
        lambda s: ARCHETYPE.get(str(s).strip(), "unknown"))
    df["label_source"] = "machine_annotator_v2"

    unknown = int((df["archetype"] == "unknown").sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info("wrote %s (%d rows)", args.output, len(df))
    logger.info("unmapped sub-categories: %d (%.1f%%)", unknown, unknown / len(df) * 100)
    logger.info("\npositive rate per label:")
    for col in LABELS:
        logger.info("  %-14s %5d  (%.1f%%)", col, df[col].sum(), df[col].mean() * 100)
    logger.info("\nmean labels per product: %.2f", df[LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()
