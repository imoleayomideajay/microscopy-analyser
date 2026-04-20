import streamlit as st


def render():
    st.markdown("# MicroAnalyser")
    st.markdown(
        "<p style='color:#6868a0; font-size:1.05rem; margin-top:-0.8rem;'>"
        "A toolkit addressing the 5 core problems in microscopy image quantification — "
        "now with deep-learning segmentation and batch processing.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Core analysis modules")

    problems = [
        {"icon": "📐", "num": "01", "title": "Scale Calibration",
         "desc": "Convert pixel measurements to physical units. Handles missing scale bars, different objectives, and camera binning."},
        {"icon": "✂️", "num": "02", "title": "Segmentation",
         "desc": "Robust thresholding with 7 methods including adaptive Sauvola/Niblack, plus rolling ball background correction."},
        {"icon": "🔗", "num": "03", "title": "Object Separation",
         "desc": "Watershed-based splitting of touching and overlapping objects. Prevents under-counting in dense samples."},
        {"icon": "🎯", "num": "04", "title": "Focus Quality",
         "desc": "Tile-based sharpness maps using Laplacian, Tenengrad, Brenner, and normalised variance metrics."},
        {"icon": "📊", "num": "05", "title": "Sampling & Edge Bias",
         "desc": "Guard frame method and Abercrombie correction for statistically unbiased counts and size estimates."},
    ]

    cols = st.columns(2)
    for i, p in enumerate(problems):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="problem-card">
                <h4>{p['icon']} &nbsp;{p['num']} — {p['title']}</h4>
                <p>{p['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### New modules")

    new_cols = st.columns(2)
    with new_cols[0]:
        st.markdown("""
        <div class="problem-card" style="border-color:#2a2a60;">
            <h4>🧬 &nbsp;Cellpose (Deep Learning)</h4>
            <p>State-of-the-art cell segmentation using the Cellpose neural network.
            Handles touching cells, variable illumination, and complex morphologies without
            manual threshold tuning. Supports <code>cyto3</code>, <code>nuclei</code>,
            <code>tissuenet</code>, and <code>livecell</code> models.</p>
        </div>
        """, unsafe_allow_html=True)

    with new_cols[1]:
        st.markdown("""
        <div class="problem-card" style="border-color:#2a2a60;">
            <h4>🗂️ &nbsp;Batch Processing</h4>
            <p>Upload an entire experiment's images and run the full pipeline on all of them.
            Results consolidate into a single CSV with per-image object counts, size statistics,
            density, and focus quality flags. Download as CSV or ZIP with pipeline parameters.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "⚡ **Caching enabled** — all heavy computations use `@st.cache_data`. "
        "After the first run, adjusting sliders reuses cached intermediates for near-instant response.",
        icon="⚡",
    )

    st.markdown("---")
    st.markdown("### Quick start")
    st.markdown("""
    1. Pick an **Analysis Module** from the sidebar
    2. **Upload** a microscopy image (TIFF, PNG, JPG)
    3. Adjust parameters with the interactive controls
    4. **Download** results as CSV or annotated image

    For multiple images from one experiment → **🗂️ Batch Processing**
    For difficult samples with touching cells → **🧬 Cellpose**
    """)

    with st.expander("🚀 Deployment options"):
        st.code("""# Streamlit Community Cloud (free)
# Push to GitHub → share.streamlit.io → New app → app.py → Deploy

# Local
pip install -r requirements.txt
streamlit run app.py

# Docker
docker build -t microanalyser .
docker run -p 8501:8501 microanalyser""", language="bash")
        st.caption("Cellpose model weights (~200 MB) are downloaded on first use and cached locally.")
