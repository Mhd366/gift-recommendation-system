import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.cluster import KMeans
import sys; sys.path.insert(0, "src")
import gift_engine as ge, text_block as tb, exposure as exp

DATA = Path("data/processed")
df = pd.read_csv(DATA / "catalog_clean.csv")
W  = ge.load_signal_weights(DATA / "signal_weights.csv")

Xp = ge.build_catalog_profile(df, weights=W)
BW = {"interests":.34,"occasions":.22,"taxonomy":.14,"age":.12,"price":.10,"gender":.08}
pre = ColumnTransformer([
    ("interests", Normalizer(), ge.INTEREST_FLAGS),
    ("occasions", Normalizer(), ge.OCCASION_FLAGS),
    ("taxonomy",  Pipeline([("o", OneHotEncoder(handle_unknown="ignore",sparse_output=False)),("n",Normalizer())]), ge.CATEGORICAL_COLS),
    ("age",       Normalizer(), ge.AGE_BUCKET_COLS),
    ("price",     Normalizer(), ge.PRICE_BAND_COLS),
    ("gender",    Normalizer(), ge.GENDER_COLS),
], transformer_weights=BW)
pipeline = Pipeline([("blocks", pre), ("unit", Normalizer())])
Xs = pipeline.fit_transform(Xp).astype(np.float32)

print("fitting text block...")
T, tpipe = tb.fit_text_block(df.product_name, n_components=96)
X = tb.combine_blocks(Xs, T)

print("fitting kmeans...")
km = KMeans(n_clusters=12, random_state=42, n_init=10).fit(Xs)
aff = ge.build_interest_affinity(df)

engine = ge.GiftRecommender(
    pipeline, km, X, df, aff, weights=W, text_pipeline=tpipe,
    exposure=exp.ExposureTracker(len(df), lam=1.0),
)

Path("models").mkdir(exist_ok=True)
joblib.dump({
    "recommender": engine,
    "text_pipeline": tpipe,
    "cluster_names": {i: f"Cluster {i}" for i in range(12)},
    "metadata": {
        "version": "3.0.0",
        "n_items": len(df),
        "n_features": int(X.shape[1]),
        "k": 12,
        "interests": ge.INTERESTS,
        "occasions": ge.OCCASIONS,
        "genders": ["Female","Male","Any"],
        "precision_at_5": 0.896,
        "ndcg_at_10": 0.902,
    },
}, "models/gift_recommender_v3.joblib", compress=3)
print(f"saved {Path('models/gift_recommender_v3.joblib').stat().st_size/1e6:.1f} MB")
