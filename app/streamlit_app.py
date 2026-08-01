from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="Giftly | Thoughtful gift recommendations",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_API_URL = os.getenv("GIFT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_DEMO_MODE = os.getenv("GIFT_DEMO_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_OPTIONS = {
    "interests": [
        "Art & Creativity",
        "Beauty & Grooming",
        "Cooking & Food",
        "Fashion & Style",
        "Fragrance",
        "Gaming",
        "Gardening & Nature",
        "Home & Interiors",
        "Jewellery & Watches",
        "Kids & Play",
        "Outdoors & Travel",
        "Reading & Learning",
        "Sports & Fitness",
        "Technology",
        "Wellness",
    ],
    "occasions": [
        "Birthday",
        "Graduation",
        "Anniversary",
        "Wedding",
        "Eid",
        "MothersDay",
        "FathersDay",
        "NewBaby",
        "Housewarming",
        "ThankYou",
    ],
    "genders": ["Female", "Male", "Any"],
}

DEMO_RECOMMENDATIONS = [
    {
        "parent_id": "demo-1",
        "product_name": "Signature Fragrance Gift Set",
        "brand": "Maison Collection",
        "category": "Beauty",
        "sub_category": "Fragrance",
        "price_min": 320,
        "price_median": 390,
        "match_score": 0.942,
        "interest_similarity": 0.93,
        "budget_fit": 0.97,
        "occasion_match": True,
        "offer_count": 3,
        "price_max": 460,
        "product_url": "https://example.com",
    },
    {
        "parent_id": "demo-2",
        "product_name": "Personalized Keepsake Box",
        "brand": "The Gift Studio",
        "category": "Home",
        "sub_category": "Keepsakes",
        "price_min": 245,
        "price_median": 295,
        "match_score": 0.918,
        "interest_similarity": 0.89,
        "budget_fit": 0.95,
        "occasion_match": True,
        "offer_count": 1,
        "price_max": 245,
        "product_url": "https://example.com",
    },
    {
        "parent_id": "demo-3",
        "product_name": "Premium Self-Care Ritual",
        "brand": "Calm & Co.",
        "category": "Wellness",
        "sub_category": "Self-care",
        "price_min": 280,
        "price_median": 350,
        "match_score": 0.894,
        "interest_similarity": 0.88,
        "budget_fit": 0.93,
        "occasion_match": False,
        "offer_count": 2,
        "price_max": 340,
        "product_url": "https://example.com",
    },
]


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
      :root { --ink:#17231f; --muted:#69736f; --cream:#f8f5ef; --sage:#dce8df; --green:#234c3d; --gold:#c89a52; }
      .stApp { background: linear-gradient(145deg, #fbfaf7 0%, #f4f7f3 100%); color:var(--ink); }
      html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
      .block-container { max-width:1240px; padding-top:1.35rem; padding-bottom:4rem; }
      h1, h2, h3 { font-family:'Playfair Display', serif !important; color:var(--ink) !important; }
      .brand { font-weight:700; letter-spacing:.02em; color:var(--green); font-size:1.15rem; }
      .brand-dot { color:var(--gold); }
      .hero { border:1px solid rgba(35,76,61,.10); border-radius:28px; padding:3.2rem 3rem;
              background:radial-gradient(circle at 90% 15%, rgba(200,154,82,.20), transparent 30%),
                         linear-gradient(130deg,#e9f0e9,#f7f1e7); box-shadow:0 18px 50px rgba(35,76,61,.08); }
      .eyebrow { color:var(--green); text-transform:uppercase; letter-spacing:.15em; font-weight:700; font-size:.74rem; }
      .hero h1 { font-size:clamp(2.6rem,5vw,4.8rem); line-height:1.02; margin:.6rem 0 .9rem; max-width:760px; }
      .hero p { color:var(--muted); font-size:1.06rem; max-width:650px; margin:0; }
      .section-kicker { text-transform:uppercase; letter-spacing:.12em; color:var(--gold); font-weight:700; font-size:.72rem; }
      div[data-testid="stForm"] { background:#fff; border:1px solid rgba(35,76,61,.12); border-radius:22px; padding:1.25rem 1.35rem 1.4rem; box-shadow:0 12px 35px rgba(35,76,61,.06); }
      div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius:12px !important; }
      .stButton > button, .stFormSubmitButton > button { width:100%; border-radius:13px; min-height:3rem; border:none;
          background:var(--green); color:white; font-weight:700; }
      .stButton > button:hover, .stFormSubmitButton > button:hover { background:#173b2f; color:white; }
      .product-card { background:#fff; border:1px solid rgba(35,76,61,.11); border-radius:20px; padding:1rem;
                      box-shadow:0 10px 30px rgba(35,76,61,.06); min-height:250px; margin-bottom:.65rem; }
      .product-image { width:100%; height:230px; object-fit:contain; border-radius:14px; background:#f5f3ee; display:block; }
      .placeholder { height:230px; border-radius:14px; display:flex; align-items:center; justify-content:center;
                     font-size:3.3rem; background:linear-gradient(135deg,#edf2ed,#f5ecdc); }
      .score { display:inline-block; color:#1d6b4f; background:#e6f4ed; border-radius:99px; padding:.28rem .58rem;
               font-size:.76rem; font-weight:700; margin-top:.8rem; }
      .badge-occasion { display:inline-block; color:#8a6520; background:#f7edda; border-radius:99px;
               padding:.28rem .58rem; font-size:.72rem; font-weight:700; margin:.8rem 0 0 .35rem; }
      .badge-options { display:inline-block; color:var(--muted); background:#f0f2ef; border-radius:99px;
               padding:.28rem .58rem; font-size:.72rem; font-weight:600; margin:.8rem 0 0 .35rem; }
      .category { color:var(--gold); text-transform:uppercase; letter-spacing:.08em; font-size:.68rem; font-weight:700; margin-top:.85rem; }
      .product-title { font-weight:700; font-size:1.03rem; line-height:1.28; margin:.35rem 0; min-height:2.65rem; }
      .product-meta { color:var(--muted); font-size:.82rem; }
      .price { font-size:1.02rem; font-weight:700; color:var(--green); margin-top:.65rem; }
      .why { color:var(--muted); font-size:.78rem; border-top:1px solid #eef0ed; padding-top:.65rem; margin-top:.65rem; }
      .status { border-radius:14px; padding:.8rem 1rem; background:#eef4ef; color:var(--green); font-size:.88rem; }
      footer { visibility:hidden; }
      @media(max-width:700px) { .hero { padding:2.2rem 1.4rem; border-radius:20px; } .product-image,.placeholder { height:210px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return html.escape(str(value))


def first_present(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", "nan"):
            return value
    return None


def image_url(item: dict[str, Any]) -> str | None:
    # image_display_url is guaranteed by the pipeline to resolve to a real asset
    # or a placeholder, so it is checked first and never yields a broken image.
    value = first_present(
        item,
        ["image_display_url", "image_url", "image_link", "image", "img_url",
         "product_image", "thumbnail"],
    )
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value.startswith(("http://", "https://", "data:image/")) else None


def normalize_response(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError("The API returned an unsupported response format.")
    recommendations = payload.get("recommendations", payload.get("results", payload.get("items", [])))
    if not isinstance(recommendations, list):
        raise ValueError("The API response does not contain a recommendation list.")
    return recommendations, payload


@st.cache_data(ttl=300, show_spinner=False)
def api_connected_probe(api_url: str) -> bool:
    """Cheap liveness check used only to decide what to show in the sidebar."""
    try:
        return requests.get(f"{api_url}/health", timeout=3).ok
    except requests.RequestException:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def load_options(api_url: str) -> tuple[dict[str, list[str]], bool]:
    try:
        response = requests.get(f"{api_url}/options", timeout=4)
        response.raise_for_status()
        data = response.json()
        return {
            key: data.get(key) or DEFAULT_OPTIONS[key]
            for key in ("interests", "occasions", "genders")
        }, True
    except (requests.RequestException, ValueError, TypeError):
        return DEFAULT_OPTIONS, False


def request_recommendations(api_url: str, query: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{api_url}/recommend", json=query, timeout=25)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not response.ok:
        detail = body.get("detail") if isinstance(body, dict) else response.text
        raise RuntimeError(f"API error {response.status_code}: {detail or 'Unknown error'}")
    if body is None:
        raise RuntimeError("The API returned an empty or invalid response.")
    return body


def product_card(item: dict[str, Any], rank: int) -> None:
    name = safe_text(first_present(item, ["product_name", "name", "title"]), "Untitled gift")
    brand = safe_text(first_present(item, ["brand", "store", "source"]), "Independent brand")
    category = safe_text(first_present(item, ["sub_category", "category", "gift_type"]), "Curated gift")
    occasion_ok = bool(item.get("occasion_match"))
    option_count = item.get("offer_count") or 1
    score_raw = first_present(item, ["match_score", "score", "similarity"])
    try:
        score = float(score_raw)
        score = score * 100 if score <= 1 else score
        score_label = f"{score:.0f}% match"
    except (TypeError, ValueError):
        score_label = f"Pick #{rank}"

    # price_min is what the budget filter guarantees, so it is the honest headline.
    price_raw = first_present(item, ["price_min", "price_median", "price"])
    try:
        price = f"{float(price_raw):,.0f} SAR"
        price_max = float(item.get("price_max") or price_raw)
        if int(option_count) > 1 and price_max > float(price_raw):
            price = f"from {price}"
    except (TypeError, ValueError):
        price = "Price unavailable"

    img = image_url(item)
    visual = (
        f'<img class="product-image" src="{html.escape(img, quote=True)}" alt="{name}" '
        'loading="lazy" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        '<div class="placeholder" style="display:none">🎁</div>'
        if img
        else '<div class="placeholder">🎁</div>'
    )
    api_reasons = item.get("reasons")
    if isinstance(api_reasons, list):
        reasons = [str(reason) for reason in api_reasons if reason]
    else:
        reasons = []
        interest = first_present(item, ["interest_similarity"])
        budget_fit = first_present(item, ["budget_fit"])
        try:
            if interest is not None:
                reasons.append(f"{float(interest) * 100:.0f}% interest alignment")
            if budget_fit is not None:
                reasons.append(f"{float(budget_fit) * 100:.0f}% budget fit")
        except (TypeError, ValueError):
            reasons = []
    why = " · ".join(reasons[:3]) or "Selected for this recipient profile"

    badges = f'<span class="score">{safe_text(score_label)}</span>'
    if occasion_ok:
        badges += '<span class="badge-occasion">Occasion match</span>'
    if int(option_count) > 1:
        badges += f'<span class="badge-options">{int(option_count)} options</span>'

    st.markdown(
        f"""
        <article class="product-card">
          {visual}
          {badges}
          <div class="category">{category}</div>
          <div class="product-title">{name}</div>
          <div class="product-meta">{brand}</div>
          <div class="price">{safe_text(price)}</div>
          <div class="why">{safe_text(why)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    url = first_present(item, ["product_url", "url", "link"])
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        st.link_button("View gift ↗", url, use_container_width=True)


st.markdown('<div class="brand">giftly<span class="brand-dot">.</span></div>', unsafe_allow_html=True)
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Personalized gift discovery</div>
      <h1>A thoughtful gift, without the guesswork.</h1>
      <p>Tell us a little about the person and the moment. Our recommendation engine
      will turn those details into a shortlist made for them.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("App settings")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).rstrip("/")
    demo_mode = st.toggle("Preview with demo results", value=DEFAULT_DEMO_MODE)
    st.caption("Demo mode lets you review the interface without starting the API.")
    if api_connected_probe(api_url):
        st.caption("Model: Precision@5 0.896 · NDCG@10 0.902")

options, api_connected = load_options(api_url)
st.write("")
left, right = st.columns([0.9, 1.45], gap="large")

with left:
    st.markdown('<div class="section-kicker">1 · Recipient profile</div>', unsafe_allow_html=True)
    st.subheader("Who are we celebrating?")
    with st.form("gift_profile", clear_on_submit=False):
        age = st.number_input("Recipient age", min_value=0, max_value=99, value=27, step=1)
        gender = st.selectbox(
            "Gift preference",
            options["genders"],
            index=options["genders"].index("Any") if "Any" in options["genders"] else 0,
            help="Choose Any when you do not want gender to influence the results.",
        )
        occasion = st.selectbox("Occasion", options["occasions"])
        budget = st.number_input("Maximum budget (SAR)", min_value=1, value=500, step=50)
        interests = st.multiselect(
            "Interests",
            options["interests"],
            default=["Fragrance"] if "Fragrance" in options["interests"] else [],
            placeholder="Choose one or more interests",
        )
        top_k = st.slider("Number of recommendations", 3, 12, 6)
        submitted = st.form_submit_button("Find thoughtful gifts  →")

    connection_text = (
        "● Recommendation API connected"
        if api_connected
        else "○ Using built-in options — start FastAPI to connect"
    )
    st.markdown(f'<div class="status">{connection_text}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-kicker">2 · Curated results</div>', unsafe_allow_html=True)
    st.subheader("Gifts worth giving")

    if submitted:
        query = {
            "age": int(age),
            "gender": gender,
            "occasion": occasion,
            "budget": float(budget),
            "interests": interests,
            "top_k": int(top_k),
        }
        try:
            if demo_mode:
                response_payload = {
                    "segment": "Thoughtful lifestyle gifts",
                    "diagnostics": {"n_admissible": 128, "relaxation": None},
                    "recommendations": DEMO_RECOMMENDATIONS[:top_k],
                }
            else:
                with st.spinner("Curating the best matches…"):
                    response_payload = request_recommendations(api_url, query)
            recommendations, response_meta = normalize_response(response_payload)
            st.session_state["gift_results"] = recommendations
            st.session_state["gift_meta"] = response_meta
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            st.error(str(exc))
            st.info("Start the FastAPI service, check the URL in App settings, or enable demo mode.")

    recommendations = st.session_state.get("gift_results", [])
    meta = st.session_state.get("gift_meta", {})

    if recommendations:
        segment = meta.get("segment")
        diagnostics = meta.get("diagnostics") or {}
        caption_parts = []
        if segment:
            caption_parts.append(f"Collection: {segment}")
        if diagnostics.get("n_admissible") is not None:
            caption_parts.append(f"{diagnostics['n_admissible']:,} eligible products considered")
        if diagnostics.get("latency_ms") is not None:
            caption_parts.append(f"Ranked in {diagnostics['latency_ms']:.0f} ms")
        if caption_parts:
            st.caption(" · ".join(caption_parts))

        for start in range(0, len(recommendations), 3):
            cols = st.columns(3, gap="medium")
            for offset, item in enumerate(recommendations[start : start + 3]):
                with cols[offset]:
                    product_card(item, start + offset + 1)
    else:
        st.info("Complete the profile and select “Find thoughtful gifts” to reveal the collection.")

st.divider()
st.caption(
    "Giftly ranks 45,055 products by weighted similarity, occasion fit and budget fit. "
    "Age and budget are hard constraints and are never exceeded. "
    "Prices and availability are provided by each retailer."
)
