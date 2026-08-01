"""Text embedding block for product names.

The catalogue's 45,055 products occupy only 2,203 distinct structural feature
vectors, because every structural field is categorical. 95% of the catalogue is
therefore geometrically indistinguishable: cosine similarity cannot separate
two products that share a category, price band, interest set and occasion set,
so the ranking within those blocks is arbitrary.

``product_name`` is the only field carrying independent information. (The
``description`` column is template-generated from the structural fields, so
embedding it would re-encode what the model already reads.)

Why character n-grams rather than a sentence transformer
--------------------------------------------------------
The catalogue is ~30% Arabic. A word-level vectoriser fitted mostly on Latin
tokens fragments Arabic badly, and an English-only sentence model would discard
that portion entirely. Character n-grams (3-5) are script-agnostic: they capture
Arabic morphology and English compounds alike, need no pretrained download, run
in seconds, and are fully reproducible from a seed — which matters more for a
reproducible pipeline than the marginal semantic gain of a large encoder.

TruncatedSVD then reduces the sparse TF-IDF matrix to a dense block that can be
concatenated with the structural blocks, L2-normalised so that cosine remains
well defined across the combined space.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")


def normalise_name(text: str) -> str:
    """Fold Arabic letter variants and strip diacritics before vectorising.

    Arabic writes the same word with several orthographic variants (أ إ آ for
    alef, ى for ya, ة for ta-marbuta). Without folding, "ساعة" and "ساعه" become
    unrelated tokens and the same product splits across two regions of the
    vector space.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = ARABIC_DIACRITICS.sub("", text)
    for src, dst in (("أإآ", "ا"), ("ى", "ي"), ("ة", "ه")):
        for ch in src:
            text = text.replace(ch, dst)
    return re.sub(r"\s+", " ", text).strip()


def build_text_pipeline(n_components: int = 96, seed: int = 42) -> Pipeline:
    """Return an unfitted char-n-gram TF-IDF to SVD to L2 pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=200_000,
            sublinear_tf=True,
        )),
        ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
        ("l2", Normalizer()),
    ])


def fit_text_block(names: pd.Series, n_components: int = 96,
                   seed: int = 42) -> tuple[np.ndarray, Pipeline]:
    """Fit the text pipeline on product names and return the dense block."""
    cleaned = names.map(normalise_name)
    pipe = build_text_pipeline(n_components, seed)
    block = pipe.fit_transform(cleaned).astype(np.float32)
    return block, pipe


def transform_text_block(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    """Project new text (a query hint) into the fitted text space."""
    return pipe.transform([normalise_name(t) for t in texts]).astype(np.float32)


def combine_blocks(structural: np.ndarray, text: np.ndarray,
                   w_structural: float = 0.70,
                   w_text: float = 0.30) -> np.ndarray:
    """Concatenate the structural and text blocks under explicit weights.

    Each block arrives unit-norm, so the weights alone decide their relative
    influence. A final L2 pass makes every row unit length, which keeps a plain
    dot product equal to cosine similarity.

    The text weight is held below the structural weight deliberately: the text
    block exists to break ties inside a structurally-matched group, not to
    override the constraint-derived signals.
    """
    combined = np.hstack([structural * w_structural, text * w_text])
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return (combined / np.clip(norms, 1e-9, None)).astype(np.float32)
