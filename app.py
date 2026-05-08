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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1f2937;
}
code, .stCode, pre {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
}

/* Main background */
.stApp {
    background: #ffffff;
}

/* Cards */
.problem-card {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.problem-card:hover {
    border-color: #2563eb;
}
.problem-card h4 {
    color: #1d4ed8;
    margin: 0 0 0.3rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.problem-card p {
    color: #334155;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.82rem;
    color: #1e3a8a;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem 0.2rem 0 0;
}

/* Headings */
h1, h2, h3 {
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #1d4ed8;
    border-color: #1e40af;
    color: #ffffff;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1.5px dashed #94a3b8;
    border-radius: 12px;
}

/* Sliders */
.stSlider > div > div > div {
    background: #2563eb !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom: 2px solid #1d4ed8 !important;
    background: transparent !important;
}

/* Divider */
hr {
    border-color: #e5e7eb !important;
}

/* Captions / small text */
.stCaption, small, [data-testid="stMarkdownContainer"] p {
    color: #475569 !important;
}

/* Success / info boxes */
.stSuccess {
    background: #ecfdf5;
    border-color: #10b981;
}
.stInfo {
    background: #eff6ff;
    border-color: #3b82f6;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MicroAnalyser")
    st.markdown("<p style='color:#475569; font-size:0.86rem; margin-top:-0.3rem;'>Microscopy image analysis toolkit</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#334155; font-size:0.78rem; letter-spacing:0.03em; text-transform:uppercase;'>Analysis modules</p>", unsafe_allow_html=True)

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
    choice = st.radio("Choose an analysis module", list(pages.keys()), label_visibility="collapsed")
    active = pages[choice]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#334155; line-height:1.6;'>
    <b style='color:#1f2937;'>About</b><br>
    Tackles the 5 core problems in<br>
    microscopy image quantification.<br><br>
    <b style='color:#1f2937;'>New</b><br>
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
