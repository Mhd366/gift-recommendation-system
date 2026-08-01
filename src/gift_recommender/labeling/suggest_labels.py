"""Generate reviewable occasion suggestions for the development annotation set.

**These are suggestions, not labels.** Every row must be read and corrected by a
human before the file is used. The output is never applied to ``gold_test.csv``.

Independence from the rule engine
---------------------------------
``configs/taxonomy.yaml`` assigns occasions from *category, gift type, age and
gender*. If this module used the same signals, reviewing its output would
merely confirm the rules against themselves - the circular evaluation this
project exists to avoid.

This module therefore reads a different signal: the **product name text**, in
Arabic and English, plus price. Name semantics are information the rule engine
never sees, so agreement between the two is evidence rather than tautology.

Usage::

    python -m gift_recommender.labeling.suggest_labels \
        --input data/annotations/gold_dev.csv \
        --output data/annotations/gold_dev_suggested.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Name lexicons. Bilingual because ~30% of product names are Arabic.
# A term implies the product TYPE, from which occasion fit is inferred.
# --------------------------------------------------------------------------
LEXICON: dict[str, list[str]] = {
    "romantic": ["necklace", "bracelet", "ring", "pendant", "perfume", "rose",
                 "diamond", "heart", "love", "خاتم", "سوار", "قلادة", "عطر", "ورد", "قلب"],
    "milestone": ["watch", "pen", "leather", "briefcase", "laptop", "tablet",
                  "headphone", "ساعة", "قلم", "حقيبة", "لابتوب", "سماعة"],
    "household": ["candle", "vase", "cushion", "dinner", "plate", "mug", "cutlery",
                  "bedding", "towel", "lamp", "frame", "bookcase", "شمعة", "مزهرية",
                  "طقم", "صحن", "كوب", "وسادة", "مصباح"],
    "child": ["lego", "toy", "puzzle", "doll", "scooter", "plush", "colouring",
              "pokemon", "roblox", "لعبة", "العاب", "بازل", "دمية", "سكوتر", "اطفال", "طفل"],
    "baby": ["baby", "infant", "newborn", "stroller", "pacifier", "bib", "nursery",
             "رضيع", "مولود", "حديثي", "عربة"],
    "beauty": ["perfume", "lipstick", "serum", "cream", "mascara", "palette", "skincare",
               "عطر", "احمر شفاه", "سيروم", "كريم", "مكياج", "عناية"],
    "edible": ["chocolate", "dates", "cake", "sweets", "honey", "coffee", "tea",
               "شوكولاتة", "تمر", "كيك", "حلوى", "عسل", "قهوة", "شاي"],
    "floral": ["bouquet", "flower", "lily", "rose", "orchid", "باقة", "ورد", "زهور"],
    "sport": ["sneaker", "jersey", "football", "gym", "fitness", "yoga", "running",
              "كرة", "تيشيرت", "رياضة", "جري", "لياقة"],
    "gaming": ["steam", "playstation", "xbox", "nintendo", "gaming", "game", "gift card",
               "controller", "لعبة", "بلايستيشن"],
    "intimate": ["lingerie", "underwear", "perfume for her", "ملابس داخلية"],
}

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")


def normalise(text: str) -> str:
    """Lowercase, strip Arabic diacritics, and unify alef/yaa/taa-marbuta forms."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = ARABIC_DIACRITICS.sub("", text)
    for src, dst in (("أإآ", "ا"), ("ى", "ي"), ("ة", "ه")):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def signals(name: str) -> set[str]:
    """Return the lexicon groups whose terms appear in a product name."""
    norm = normalise(name)
    return {group for group, terms in LEXICON.items()
            if any(normalise(t) in norm for t in terms)}


def suggest(name: str, price: float) -> dict[str, int]:
    """Suggest binary occasion labels from the product name and price.

    Args:
        name: Raw product name, Arabic or English.
        price: Price in SAR. Gates occasions with social price expectations.

    Returns:
        A mapping from occasion label to 0 or 1.
    """
    sig = signals(name)
    cheap, mid, dear = price < 100, 100 <= price <= 2000, price > 2000

    out = {
        # Birthday is genuinely near-universal; only intimate items are excluded.
        "Birthday": int("intimate" not in sig),

        "Graduation": int(bool({"milestone", "gaming"} & sig) and not cheap
                          and not ({"child", "baby"} & sig)),

        "Anniversary": int(bool({"romantic", "beauty", "floral"} & sig) and not cheap
                           and not ({"child", "baby"} & sig)),

        "Wedding": int(bool({"household", "romantic"} & sig) and (mid or dear)
                       and not ({"child", "baby", "gaming"} & sig)),

        # Eid spans every age and price point in the Gulf.
        "Eid": int(bool({"beauty", "edible", "child", "romantic", "sport"} & sig)),

        "MothersDay": int(bool({"beauty", "floral", "edible", "romantic"} & sig)
                          and not ({"child", "baby", "gaming"} & sig)),

        "FathersDay": int(bool({"milestone", "sport", "gaming", "edible"} & sig)
                          and not ({"child", "baby", "beauty"} & sig)),

        "NewBaby": int(bool({"baby"} & sig)
                       or (bool({"floral", "beauty"} & sig) and not dear)),

        "Housewarming": int(bool({"household", "floral"} & sig) and not dear),

        # A thank-you gift must be neither expensive nor intimate.
        "ThankYou": int(bool({"edible", "floral", "household", "beauty"} & sig)
                        and price <= 500 and "intimate" not in sig),
    }
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Suggest occasion labels for review.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    suggestions = pd.DataFrame(
        [suggest(r["Product Name"], r["Price (SAR)"]) for _, r in df.iterrows()],
        index=df.index,
    )
    for col in suggestions.columns:
        df[col] = suggestions[col]

    # A row matched by no lexicon group is a low-confidence guess: flag it so the
    # reviewer starts where the machine is weakest.
    df["review_priority"] = [
        "HIGH - no name signal" if not signals(n) else "normal"
        for n in df["Product Name"]
    ]
    df["source"] = "SUGGESTED - review required"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info("wrote %s (%d rows)", args.output, len(df))
    logger.info("\nsuggested positive rate per label:")
    for col in suggestions.columns:
        logger.info("  %-14s %5d  (%.1f%%)", col, df[col].sum(), df[col].mean() * 100)
    high = int((df["review_priority"].str.startswith("HIGH")).sum())
    logger.info("\n%d rows flagged HIGH priority (%.1f%%) - review these first",
                high, high / len(df) * 100)


if __name__ == "__main__":
    main()
