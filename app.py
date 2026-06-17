# app.py  —  SentimentScope  (UI v2)
# Run:  streamlit run app.py

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import clean_text, label_to_display

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentScope",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS  ──────────────────────────────────────────────────────────────────────
# Design tokens
# Background : #0D0D14  (near-black, not pure black)
# Surface     : #16161F  cards
# Border      : #2A2A3D
# Accent      : #7C5CFC  purple-violet
# Text-primary: #F0F0FF  (off-white, not #fff)
# Text-muted  : #8B8BA7
# Positive    : #22C55E
# Neutral     : #FBBF24
# Negative    : #F43F5E
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #0D0D14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #F0F0FF !important;
}

/* Wipe Streamlit's own background layers */
[data-testid="stHeader"]          { background: transparent !important; }
[data-testid="stToolbar"]         { display: none !important; }
#MainMenu, footer                 { visibility: hidden !important; }
[data-testid="stDecoration"]      { display: none !important; }
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ── Typography  ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #F0F0FF !important;
    font-family: 'Inter', sans-serif !important;
}

p, li, span, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #C8C8E0 !important;
    line-height: 1.65 !important;
}

/* ── Cards ── */
.ss-card {
    background: #16161F;
    border: 1px solid #2A2A3D;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}
.ss-card-accent {
    background: linear-gradient(135deg, #1A1530 0%, #16161F 60%);
    border: 1px solid #3D2F7A;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}

/* ── Hero ── */
.hero-wrap {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1E1E2E;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(90deg, #A78BFA 0%, #7C5CFC 40%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8B8BA7 !important;
    font-weight: 400;
    max-width: 540px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #1A1530;
    border: 1px solid #3D2F7A;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #A78BFA !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section labels ── */
.ss-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8B8BA7 !important;
    margin-bottom: 0.5rem;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: #1C1C28 !important;
    border: 1.5px solid #2A2A3D !important;
    border-radius: 12px !important;
    color: #F0F0FF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.97rem !important;
    line-height: 1.65 !important;
    padding: 1rem !important;
    caret-color: #7C5CFC !important;
    transition: border-color 0.2s !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #7C5CFC !important;
    box-shadow: 0 0 0 3px rgba(124,92,252,0.18) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #4A4A6A !important;
}
/* Hide the default label Streamlit renders */
.stTextArea label { display: none !important; }

/* ── Buttons ── */
/* Primary */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.25rem !important;
    transition: all 0.18s ease !important;
    border: none !important;
    width: 100% !important;
    cursor: pointer !important;
}
button[kind="primary"], .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C5CFC, #5B8DEF) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(124,92,252,0.35) !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(124,92,252,0.55) !important;
    transform: translateY(-1px) !important;
}

/* Secondary / example buttons */
button[kind="secondary"], .stButton > button[kind="secondary"] {
    background: #1C1C28 !important;
    color: #C8C8E0 !important;
    border: 1px solid #2A2A3D !important;
}
button[kind="secondary"]:hover {
    background: #22223A !important;
    border-color: #7C5CFC !important;
    color: #F0F0FF !important;
}

/* ── Result badge ── */
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.5rem;
    border-radius: 999px;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 1.25rem;
}

/* ── Probability bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    width: 72px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #C8C8E0;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    height: 8px;
    background: #22223A;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}
.prob-pct {
    width: 38px;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Confidence arc (big number) ── */
.conf-big {
    font-size: 4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
}
.conf-unit {
    font-size: 1.4rem;
    font-weight: 700;
    vertical-align: super;
}
.conf-track {
    width: 100%;
    height: 6px;
    background: #22223A;
    border-radius: 999px;
    margin-top: 0.75rem;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 3.5rem 1rem;
    text-align: center;
}
.empty-icon { font-size: 2.8rem; opacity: 0.5; }
.empty-txt  { color: #4A4A6A !important; font-size: 0.9rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.4rem !important;
    border-bottom: 1px solid #1E1E2E !important;
    padding-bottom: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #8B8BA7 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1rem !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #F0F0FF !important; }
.stTabs [aria-selected="true"] {
    color: #A78BFA !important;
    border-bottom-color: #7C5CFC !important;
    background: transparent !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1C1C28 !important;
    border: 1.5px dashed #2A2A3D !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: #C8C8E0 !important;
}
[data-testid="stFileUploader"] small {
    color: #8B8BA7 !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #1C1C28 !important;
    border: 1.5px solid #2A2A3D !important;
    border-radius: 10px !important;
    color: #F0F0FF !important;
}
[data-baseweb="select"] span { color: #F0F0FF !important; }
[data-baseweb="popover"] { background: #1C1C28 !important; }
[data-baseweb="menu-item"] { color: #F0F0FF !important; }
[data-baseweb="menu-item"]:hover { background: #22223A !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2A2A3D !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.dvn-scroller { background: #16161F !important; }
[data-testid="stDataFrame"] th {
    background: #1C1C28 !important;
    color: #8B8BA7 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stDataFrame"] td { color: #F0F0FF !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1C1C28 !important;
    border: 1px solid #2A2A3D !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #C8C8E0 !important; }
[data-testid="stExpander"] summary:hover { color: #F0F0FF !important; }

/* ── Code block ── */
.stCode, code, pre {
    background: #0A0A12 !important;
    color: #A78BFA !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #2A2A3D !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #16161F !important;
    color: #A78BFA !important;
    border: 1px solid #3D2F7A !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1A1530 !important;
    border-color: #7C5CFC !important;
}

/* ── Alert / info / warning ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
}

/* ── Sidebar  ── */
[data-testid="stSidebar"] {
    background: #0D0D14 !important;
    border-right: 1px solid #1E1E2E !important;
}
[data-testid="stSidebar"] * { color: #C8C8E0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #7C5CFC !important; }

/* ── Divider ── */
hr { border-color: #1E1E2E !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

model, vectorizer = load_model()

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS = {0: "#F43F5E", 1: "#FBBF24", 2: "#22C55E"}
LABEL_EMOJIS = {0: "😞", 1: "😐", 2: "😊"}


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, None
    vec   = vectorizer.transform([cleaned])
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    return label, float(proba[label]), proba


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🧠 ML-Powered · Free · Local</div>
  <div class="hero-title">SentimentScope</div>
  <div class="hero-sub">Paste any text — review, tweet, or feedback — and get instant sentiment analysis with confidence scores.</div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("⚠️ **Model not found.** Run `python train_model.py` first, then refresh.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✍️  Analyse Text", "📂  Batch Upload", "📊  Model Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.05, 1], gap="large")

    # ── Left: Input ──────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="ss-label">Your Text</div>', unsafe_allow_html=True)

        user_text = st.text_area(
            label="text_input",
            placeholder="Type or paste a movie review, tweet, or any text here…",
            height=180,
            key="single_input",
        )

        # Example buttons
        st.markdown(
            '<div class="ss-label" style="margin-top:1rem;margin-bottom:0.4rem">Try an example</div>',
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3)
        examples = {
            "😊 Positive": "Outstanding film! The performances were breathtaking and the story left me in awe. Absolutely loved it.",
            "😞 Negative": "Terrible waste of time. Boring plot, wooden acting, and a dreadful ending. Would not recommend.",
            "😐 Neutral":  "The movie had some good moments, but the pacing felt uneven. A decent watch, nothing more.",
        }
        for btn_label, col in zip(examples, [c1, c2, c3]):
            if col.button(btn_label, use_container_width=True, key=f"ex_{btn_label}"):
                st.session_state["single_input"] = examples[btn_label]
                st.rerun()

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍  Analyse Sentiment", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # How it works — small info card below input
        st.markdown("""
<div class="ss-card" style="padding:1.2rem 1.5rem">
  <div class="ss-label">How it works</div>
  <div style="display:flex;gap:1.5rem;margin-top:0.6rem;flex-wrap:wrap">
    <div style="flex:1;min-width:100px">
      <div style="font-size:1.2rem">🧹</div>
      <div style="font-size:0.78rem;font-weight:700;color:#F0F0FF;margin:0.2rem 0 0.1rem">Clean</div>
      <div style="font-size:0.75rem;color:#8B8BA7;line-height:1.4">Remove noise, stopwords, punctuation</div>
    </div>
    <div style="flex:1;min-width:100px">
      <div style="font-size:1.2rem">📐</div>
      <div style="font-size:0.78rem;font-weight:700;color:#F0F0FF;margin:0.2rem 0 0.1rem">Vectorise</div>
      <div style="font-size:0.75rem;color:#8B8BA7;line-height:1.4">TF-IDF with uni + bigrams</div>
    </div>
    <div style="flex:1;min-width:100px">
      <div style="font-size:1.2rem">🤖</div>
      <div style="font-size:0.78rem;font-weight:700;color:#F0F0FF;margin:0.2rem 0 0.1rem">Predict</div>
      <div style="font-size:0.75rem;color:#8B8BA7;line-height:1.4">Logistic Regression model</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Right: Result ─────────────────────────────────────────────────────────
    with right:
        if analyse_btn and user_text.strip():
            label, conf, proba = predict(user_text)

            if label is None:
                st.warning("Text became empty after cleaning — try a longer input.")
            else:
                color = LABEL_COLORS[label]
                emoji = LABEL_EMOJIS[label]
                name  = LABEL_NAMES[label]
                pct   = int(conf * 100)

                # ── Result badge ─────────────────────────────────────────────
                st.markdown(f"""
<div class="ss-card-accent">
  <div class="ss-label">Prediction</div>
  <div class="result-pill" style="background:{color}1A;border:2px solid {color}55;color:{color};margin-top:0.6rem">
    <span>{emoji}</span><span>{name}</span>
  </div>

  <div class="ss-label" style="margin-top:1rem">Confidence</div>
  <div style="display:flex;align-items:baseline;gap:0.25rem;margin-top:0.3rem">
    <span class="conf-big" style="color:{color}">{pct}</span>
    <span class="conf-unit" style="color:{color}">%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}99,{color})"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                # ── Per-class probability bars ────────────────────────────────
                st.markdown('<div class="ss-card">', unsafe_allow_html=True)
                st.markdown('<div class="ss-label">Class Probabilities</div>', unsafe_allow_html=True)

                classes = list(model.classes_)
                for cls, p in zip(classes, proba):
                    cls_int = int(cls)
                    c_name  = LABEL_NAMES.get(cls_int, str(cls_int))
                    c_color = LABEL_COLORS.get(cls_int, "#8B8BA7")
                    c_emoji = LABEL_EMOJIS.get(cls_int, "")
                    p_pct   = int(p * 100)
                    bar_w   = max(p_pct, 2)
                    st.markdown(f"""
<div class="prob-row">
  <div class="prob-label">{c_emoji} {c_name}</div>
  <div class="prob-track">
    <div class="prob-fill" style="width:{bar_w}%;background:{c_color}"></div>
  </div>
  <div class="prob-pct" style="color:{c_color}">{p_pct}%</div>
</div>
""", unsafe_allow_html=True)

                # ── Cleaned text preview ──────────────────────────────────────
                with st.expander("🔎 View cleaned text"):
                    st.code(clean_text(user_text), language=None)

                st.markdown("</div>", unsafe_allow_html=True)

        elif analyse_btn:
            st.warning("Please enter some text before analysing.")
        else:
            st.markdown("""
<div class="ss-card">
  <div class="empty-state">
    <div class="empty-icon">💬</div>
    <div class="empty-txt">Enter text on the left and click<br><strong style="color:#7C5CFC">Analyse Sentiment</strong> to see results here.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="ss-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;font-weight:700;color:#F0F0FF;margin-bottom:0.4rem">Bulk Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.88rem;color:#8B8BA7;margin-bottom:1.2rem">Upload a CSV file with a text column. SentimentScope will predict sentiment for every row.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown(
                f'<div style="font-size:0.85rem;color:#8B8BA7;margin-bottom:0.6rem">'
                f'Detected <strong style="color:#F0F0FF">{len(batch_df):,} rows</strong> · '
                f'Columns: {", ".join(f"<code style=\'color:#A78BFA\'>{c}</code>" for c in batch_df.columns)}'
                f'</div>',
                unsafe_allow_html=True
            )

            text_col = st.selectbox("Select the text column", options=batch_df.columns.tolist())

            if st.button("🚀  Run Batch Analysis", type="primary"):
                with st.spinner("Analysing all rows…"):
                    texts, results = batch_df[text_col].astype(str).tolist(), []
                    for t in texts:
                        lbl, conf, proba = predict(t)
                        if lbl is None:
                            results.append(("Unknown", 0.0, {}, ""))
                        else:
                            pd_ = {f"prob_{LABEL_NAMES.get(int(c),'unk').lower()[:3]}": round(float(p), 4)
                                   for c, p in zip(model.classes_, proba)}
                            results.append((LABEL_NAMES[lbl], round(conf, 4), pd_, clean_text(t)))

                out_df = batch_df.copy()
                out_df["sentiment"]  = [r[0] for r in results]
                out_df["confidence"] = [r[1] for r in results]
                for k in results[0][2]:
                    out_df[k] = [r[2].get(k, 0.0) for r in results]
                out_df["clean_text"] = [r[3] for r in results]

                st.success(f"✅ Analysed {len(out_df):,} rows successfully.")

                # Stats row
                vc = out_df["sentiment"].value_counts()
                col_stats = st.columns(len(vc))
                color_map = {"Positive": "#22C55E", "Negative": "#F43F5E", "Neutral": "#FBBF24", "Unknown": "#8B8BA7"}
                for i, (sent, cnt) in enumerate(vc.items()):
                    col_stats[i].markdown(
                        f'<div style="background:#16161F;border:1px solid #2A2A3D;border-radius:12px;'
                        f'padding:1rem;text-align:center">'
                        f'<div style="font-size:1.8rem;font-weight:800;color:{color_map.get(sent,"#F0F0FF")}">{cnt}</div>'
                        f'<div style="font-size:0.75rem;color:#8B8BA7;font-weight:600;text-transform:uppercase;'
                        f'letter-spacing:0.08em;margin-top:0.2rem">{sent}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Bar chart
                fig, ax = plt.subplots(figsize=(7, 3))
                bar_colors = [color_map.get(s, "#8B8BA7") for s in vc.index]
                bars = ax.bar(vc.index, vc.values, color=bar_colors, width=0.5,
                              edgecolor="none", zorder=3)
                ax.bar_label(bars, padding=5, color="#F0F0FF", fontsize=11, fontweight="bold")
                ax.set_facecolor("#16161F")
                fig.patch.set_facecolor("#16161F")
                ax.grid(axis="y", color="#22223A", linewidth=0.8, zorder=0)
                ax.tick_params(colors="#8B8BA7", labelsize=10)
                ax.spines[:].set_visible(False)
                ax.set_title("Sentiment Distribution", color="#F0F0FF", fontsize=12,
                             fontweight="700", pad=12, loc="left")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.dataframe(out_df.head(50), use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Results as CSV",
                    data=csv_bytes,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    CM_PATH   = os.path.join("model", "confusion_matrix.png")
    DATA_PATH = os.path.join("data", "dataset.csv")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="ss-label">Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, use_column_width=True)
        else:
            st.markdown('<div class="empty-state"><div class="empty-icon">📊</div><div class="empty-txt">Run <code>python train_model.py</code> to generate the confusion matrix.</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ss-card">', unsafe_allow_html=True)
        st.markdown('<div class="ss-label">How to Read</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="margin-top:0.6rem">
  <div style="margin-bottom:0.9rem">
    <div style="font-weight:700;color:#F0F0FF;font-size:0.88rem">Rows = Actual Labels</div>
    <div style="color:#8B8BA7;font-size:0.82rem;margin-top:0.2rem">What the review really was</div>
  </div>
  <div style="margin-bottom:0.9rem">
    <div style="font-weight:700;color:#F0F0FF;font-size:0.88rem">Columns = Predicted Labels</div>
    <div style="color:#8B8BA7;font-size:0.82rem;margin-top:0.2rem">What the model guessed</div>
  </div>
  <div style="margin-bottom:0.9rem">
    <div style="font-weight:700;color:#22C55E;font-size:0.88rem">✅ Diagonal = Correct</div>
    <div style="color:#8B8BA7;font-size:0.82rem;margin-top:0.2rem">High diagonal = great model</div>
  </div>
  <div>
    <div style="font-weight:700;color:#F43F5E;font-size:0.88rem">❌ Off-diagonal = Errors</div>
    <div style="color:#8B8BA7;font-size:0.82rem;margin-top:0.2rem">Where the model got confused</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ss-card" style="margin-top:0">', unsafe_allow_html=True)
        st.markdown('<div class="ss-label">ML Stack</div>', unsafe_allow_html=True)
        stack = [
            ("📐", "TF-IDF Vectoriser", "50k features · unigrams + bigrams"),
            ("🤖", "Logistic Regression", "C=1.0 · lbfgs solver · 500 iter"),
            ("🔤", "NLTK Stopwords", "English stopwords, negations kept"),
        ]
        for icon, title, detail in stack:
            st.markdown(f"""
<div style="display:flex;gap:0.75rem;align-items:flex-start;margin-bottom:0.8rem">
  <div style="font-size:1.2rem;flex-shrink:0">{icon}</div>
  <div>
    <div style="font-weight:700;color:#F0F0FF;font-size:0.85rem">{title}</div>
    <div style="color:#8B8BA7;font-size:0.78rem;margin-top:0.1rem">{detail}</div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if os.path.exists(DATA_PATH):
        st.markdown('<div class="ss-card" style="margin-top:0">', unsafe_allow_html=True)
        st.markdown('<div class="ss-label">Training Data Sample</div>', unsafe_allow_html=True)
        try:
            sample = pd.read_csv(DATA_PATH, nrows=5000, encoding="latin-1")
            st.dataframe(sample.head(8), use_container_width=True)
        except Exception:
            st.info("Could not load dataset preview.")
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;border-top:1px solid #1E1E2E;margin-top:2rem">
  <span style="color:#4A4A6A;font-size:0.78rem">
    SentimentScope &nbsp;·&nbsp; Built with Streamlit + Scikit-learn &nbsp;·&nbsp; 100% free &amp; open-source
  </span>
</div>
""", unsafe_allow_html=True)
