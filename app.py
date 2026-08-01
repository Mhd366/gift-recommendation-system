"""Giftly — Streamlit app for Hugging Face Spaces.

Loads the recommender artifact directly instead of calling a separate FastAPI
process, because Spaces runs a single process.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Giftly | Thoughtful gift recommendations",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (identical to the original design) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
  :root{--ink:#17231f;--muted:#69736f;--cream:#f8f5ef;--sage:#dce8df;--green:#234c3d;--gold:#c89a52;}
  .stApp{background:linear-gradient(145deg,#fbfaf7 0%,#f4f7f3 100%);color:var(--ink);}
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
  .block-container{max-width:1240px;padding-top:1.35rem;padding-bottom:4rem;}
  h1,h2,h3{font-family:'Playfair Display',serif !important;color:var(--ink) !important;}
  .brand{font-weight:700;letter-spacing:.02em;color:var(--green);font-size:1.15rem;}
  .brand-dot{color:var(--gold);}
  .hero{border:1px solid rgba(35,76,61,.10);border-radius:28px;padding:3.2rem 3rem;
        background:radial-gradient(circle at 90% 15%,rgba(200,154,82,.20),transparent 30%),
                   linear-gradient(130deg,#e9f0e9,#f7f1e7);box-shadow:0 18px 50px rgba(35,76,61,.08);}
  .eyebrow{color:var(--green);text-transform:uppercase;letter-spacing:.15em;font-weight:700;font-size:.74rem;}
  .hero h1{font-size:clamp(2.6rem,5vw,4.8rem);line-height:1.02;margin:.6rem 0 .9rem;max-width:760px;}
  .hero p{color:var(--muted);font-size:1.06rem;max-width:650px;margin:0;}
  .section-kicker{text-transform:uppercase;letter-spacing:.12em;color:var(--gold);font-weight:700;font-size:.72rem;}
  div[data-testid="stForm"]{background:#fff;border:1px solid rgba(35,76,61,.12);border-radius:22px;
        padding:1.25rem 1.35rem 1.4rem;box-shadow:0 12px 35px rgba(35,76,61,.06);}
  .stButton>button,.stFormSubmitButton>button{width:100%;border-radius:13px;min-height:3rem;border:none;
        background:var(--green);color:white;font-weight:700;}
  .stButton>button:hover,.stFormSubmitButton>button:hover{background:#173b2f;color:white;}
  .product-card{background:#fff;border:1px solid rgba(35,76,61,.11);border-radius:20px;padding:1rem;
                box-shadow:0 10px 30px rgba(35,76,61,.06);min-height:250px;margin-bottom:.65rem;}
  .product-image{width:100%;height:230px;object-fit:contain;border-radius:14px;background:#f5f3ee;display:block;}
  .placeholder{height:230px;border-radius:14px;display:flex;align-items:center;justify-content:center;
               font-size:3.3rem;background:linear-gradient(135deg,#edf2ed,#f5ecdc);}
  .score{display:inline-block;color:#1d6b4f;background:#e6f4ed;border-radius:99px;
         padding:.28rem .58rem;font-size:.76rem;font-weight:700;margin-top:.8rem;}
  .badge-occasion{display:inline-block;color:#8a6520;background:#f7edda;border-radius:99px;
         padding:.28rem .58rem;font-size:.72rem;font-weight:700;margin:.8rem 0 0 .35rem;}
  .badge-options{display:inline-block;color:var(--muted);background:#f0f2ef;border-radius:99px;
         padding:.28rem .58rem;font-size:.72rem;font-weight:600;margin:.8rem 0 0 .35rem;}
  .category{color:var(--gold);text-transform:uppercase;letter-spacing:.08em;font-size:.68rem;font-weight:700;margin-top:.85rem;}
  .product-title{font-weight:700;font-size:1.03rem;line-height:1.28;margin:.35rem 0;min-height:2.65rem;}
  .product-meta{color:var(--muted);font-size:.82rem;}
  .price{font-size:1.02rem;font-weight:700;color:var(--green);margin-top:.65rem;}
  .why{color:var(--muted);font-size:.78rem;border-top:1px solid #eef0ed;padding-top:.65rem;margin-top:.65rem;}
  .status{border-radius:14px;padding:.8rem 1rem;background:#eef4ef;color:var(--green);font-size:.88rem;}
  footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── load engine ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine…")
def load_engine():
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))

    import joblib
    artifact = joblib.load(Path("/mount/src/gift-recommendation-system/models/gift_recommender_v3.joblib"))
    return artifact["recommender"], artifact["metadata"], artifact.get("cluster_names", {})


try:
    ENGINE, META, CLUSTER_NAMES = load_engine()
    engine_ok = True
except Exception as e:
    engine_ok = False
    engine_error = str(e)

OPTIONS = {
    "interests": META["interests"] if engine_ok else [
        "Art & Creativity","Beauty & Grooming","Cooking & Food","Fashion & Style",
        "Fragrance","Gaming","Gardening & Nature","Home & Interiors",
        "Jewellery & Watches","Kids & Play","Outdoors & Travel","Reading & Learning",
        "Sports & Fitness","Technology","Wellness"],
    "occasions": META["occasions"] if engine_ok else [
        "Birthday","Graduation","Anniversary","Wedding","Eid",
        "MothersDay","FathersDay","NewBaby","Housewarming","ThankYou"],
    "genders": ["Female","Male","Any"],
}


# ── helpers ───────────────────────────────────────────────────────────────────
def safe(v: Any, fallback: str = "") -> str:
    return fallback if v is None else html.escape(str(v))

def first(*keys, item):
    for k in keys:
        v = item.get(k)
        if v not in (None, "", "nan"):
            return v
    return None

def product_card(item: dict, rank: int) -> None:
    name     = safe(first("product_name","name","title", item=item), "Untitled gift")
    brand    = safe(first("brand","store", item=item), "Independent brand")
    category = safe(first("sub_category","category", item=item), "Curated gift")

    score_raw = first("match_score","score", item=item)
    try:
        sc = float(score_raw)
        sc = sc * 100 if sc <= 1 else sc
        score_label = f"{sc:.0f}% match"
    except (TypeError, ValueError):
        score_label = f"Pick #{rank}"

    price_raw = first("price_min","price_median","price", item=item)
    try:
        price = f"{float(price_raw):,.0f} SAR"
        price_max = float(item.get("price_max") or price_raw)
        if int(item.get("offer_count",1)) > 1 and price_max > float(price_raw):
            price = f"from {price}"
    except (TypeError, ValueError):
        price = "Price unavailable"

    img = first("image_display_url","image_url","image", item=item)
    if isinstance(img, str) and not img.startswith(("http://","https://")):
        img = None
    visual = (
        f'<img class="product-image" src="{html.escape(img,quote=True)}" alt="{name}" '
        'loading="lazy" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        '<div class="placeholder" style="display:none">🎁</div>'
        if img else '<div class="placeholder">🎁</div>'
    )

    badges = f'<span class="score">{safe(score_label)}</span>'
    if item.get("occasion_match"):
        badges += '<span class="badge-occasion">Occasion match</span>'
    if int(item.get("offer_count", 1)) > 1:
        badges += f'<span class="badge-options">{int(item["offer_count"])} options</span>'

    reasons = item.get("reasons") or []
    why = " · ".join(str(r) for r in reasons[:3]) or "Selected for this recipient profile"

    st.markdown(f"""
    <article class="product-card">
      {visual}
      {badges}
      <div class="category">{category}</div>
      <div class="product-title">{name}</div>
      <div class="product-meta">{brand}</div>
      <div class="price">{safe(price)}</div>
      <div class="why">{safe(why)}</div>
    </article>
    """, unsafe_allow_html=True)

    url = first("product_url","url","link", item=item)
    if isinstance(url, str) and url.startswith(("http://","https://")):
        st.link_button("View gift ↗", url, use_container_width=True)


# ── page ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="brand">giftly<span class="brand-dot">.</span></div>',
            unsafe_allow_html=True)
st.markdown("""
<section class="hero">
  <div class="eyebrow">Personalized gift discovery</div>
  <h1>A thoughtful gift, without the guesswork.</h1>
  <p>Tell us a little about the person and the moment. Our recommendation engine
  will turn those details into a shortlist made for them.</p>
</section>
""", unsafe_allow_html=True)

st.write("")
left, right = st.columns([0.9, 1.45], gap="large")

with left:
    st.markdown('<div class="section-kicker">1 · Recipient profile</div>',
                unsafe_allow_html=True)
    st.subheader("Who are we celebrating?")

    with st.form("gift_profile", clear_on_submit=False):
        age      = st.number_input("Recipient age", min_value=0, max_value=99, value=27, step=1)
        gender   = st.selectbox("Gift preference", OPTIONS["genders"],
                                index=OPTIONS["genders"].index("Any"))
        occasion = st.selectbox("Occasion", OPTIONS["occasions"])
        budget   = st.number_input("Maximum budget (SAR)", min_value=1, value=500, step=50)
        interests= st.multiselect("Interests", OPTIONS["interests"],
                                  default=["Fragrance"] if "Fragrance" in OPTIONS["interests"] else [],
                                  placeholder="Choose one or more interests")
        top_k    = st.slider("Number of recommendations", 3, 12, 6)
        submitted= st.form_submit_button("Find thoughtful gifts  →")

    status = "● Engine loaded — Precision@5 0.896 · NDCG@10 0.902" if engine_ok \
             else f"○ Engine not loaded: {engine_error}"
    st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-kicker">2 · Curated results</div>',
                unsafe_allow_html=True)
    st.subheader("Gifts worth giving")

    if submitted:
        if not engine_ok:
            st.error("Engine not loaded. Check that models/gift_recommender_v3.joblib exists.")
        else:
            with st.spinner("Curating the best matches…"):
                recs, diag = ENGINE.recommend(
                    age=int(age), gender=gender, occasion=occasion,
                    budget=float(budget), interests=interests, top_k=int(top_k),
                )
            items = recs.to_dict(orient="records")
            st.session_state["gift_results"] = items
            st.session_state["gift_meta"]    = diag

    items = st.session_state.get("gift_results", [])
    diag  = st.session_state.get("gift_meta", {})

    if items:
        parts = []
        if diag.get("n_admissible"):
            parts.append(f"{diag['n_admissible']:,} eligible products considered")
        if parts:
            st.caption(" · ".join(parts))
        for start in range(0, len(items), 3):
            cols = st.columns(3, gap="medium")
            for offset, item in enumerate(items[start:start+3]):
                with cols[offset]:
                    product_card(item, start + offset + 1)
    else:
        st.info('Complete the profile and select "Find thoughtful gifts" to reveal the collection.')

st.divider()
st.caption(
    "Giftly ranks 45,055 products by weighted similarity, occasion fit and budget fit. "
    "Age and budget are hard constraints and are never exceeded. "
    "Prices and availability are provided by each retailer."
)
