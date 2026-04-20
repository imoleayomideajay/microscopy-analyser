import streamlit as st

st.set_page_config(
    page_title="MicroAnalyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .stCode, pre {
    font-family: 'DM Mono', monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * {
    color: #c8c8d8 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem;
}

/* Main background */
.stApp {
    background: #06060c;
}

/* Cards */
.problem-card {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.problem-card:hover {
    border-color: #3d3d6e;
}
.problem-card h4 {
    color: #a0a8ff;
    margin: 0 0 0.3rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.problem-card p {
    color: #7878a0;
    font-size: 0.82rem;
    margin: 0;
    line-height: 1.5;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: #13132a;
    border: 1px solid #2a2a50;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.78rem;
    color: #8888cc;
    font-family: 'DM Mono', monospace;
    margin: 0.2rem 0.2rem 0 0;
}

/* Headings */
h1, h2, h3 {
    color: #e8e8ff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button {
    background: #1e1e3f;
    color: #a0a8ff;
    border: 1px solid #3030608c;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #2a2a5e;
    border-color: #6060c0;
    color: #d0d4ff;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border: 1.5px dashed #2a2a50;
    border-radius: 12px;
}

/* Sliders */
.stSlider > div > div > div {
    background: #a0a8ff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6868a0;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #a0a8ff !important;
    border-bottom: 2px solid #a0a8ff !important;
    background: transparent !important;
}

/* Divider */
hr {
    border-color: #1e1e35 !important;
}

/* Captions / small text */
.stCaption, small, [data-testid="stMarkdownContainer"] p {
    color: #6868a0 !important;
}

/* Success / info boxes */
.stSuccess {
    background: #0a1a0a;
    border-color: #2a602a;
}
.stInfo {
    background: #0a0a1a;
    border-color: #2a2a6a;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MicroAnalyser")
    st.markdown("<p style='color:#4a4a78; font-size:0.78rem; margin-top:-0.5rem;'>Microscopy image analysis toolkit</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#4a4a78; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;'>Analysis Modules</p>", unsafe_allow_html=True)

    pages = {
        "🏠  Home": "home",
        "📐  Scale Calibration": "scale",
        "✂️  Segmentation": "segmentation",
        "🔗  Object Separation": "separation",
        "🎯  Focus & Depth": "focus",
        "📊  Sampling & Edge Bias": "sampling",
        "🧬  Cellpose (DL)": "cellpose",
        "🗂️  Batch Processing": "batch",
    }
    choice = st.radio("", list(pages.keys()), label_visibility="collapsed")
    active = pages[choice]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#3a3a60; line-height:1.7;'>
    <b style='color:#4a4a80;'>About</b><br>
    Tackles the 5 core problems in<br>
    microscopy image quantification.<br><br>
    <b style='color:#4a4a80;'>New</b><br>
    🧬 Cellpose deep-learning seg.<br>
    🗂️ Batch pipeline (multi-image)<br>
    ⚡ All modules cached for speed
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────
if active == "home":
    from pages import home
    home.render()
elif active == "scale":
    from pages import scale_calibration
    scale_calibration.render()
elif active == "segmentation":
    from pages import segmentation
    segmentation.render()
elif active == "separation":
    from pages import object_separation
    object_separation.render()
elif active == "focus":
    from pages import focus_quality
    focus_quality.render()
elif active == "sampling":
    from pages import sampling_bias
    sampling_bias.render()
elif active == "cellpose":
    from pages import cellpose_seg
    cellpose_seg.render()
elif active == "batch":
    from pages import batch_processing
    batch_processing.render()
