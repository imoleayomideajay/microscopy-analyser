import streamlit as st

def render():
    st.markdown("# MicroAnalyser")
    st.markdown("<p style='color:#6868a0; font-size:1.05rem; margin-top:-0.8rem;'>A toolkit addressing the 5 core problems in microscopy image quantification</p>", unsafe_allow_html=True)
    st.markdown("---")

    problems = [
        {
            "num": "01",
            "title": "Scale Calibration",
            "icon": "📐",
            "desc": "Convert pixel measurements to physical units with microscope-aware calibration. Handles missing scale bars, different objectives, and camera binning.",
            "module": "📐  Scale Calibration",
        },
        {
            "num": "02",
            "title": "Image Segmentation",
            "icon": "✂️",
            "desc": "Robust thresholding and segmentation with adaptive methods, background correction, and noise filtering for low-contrast images.",
            "module": "✂️  Segmentation",
        },
        {
            "num": "03",
            "title": "Object Separation",
            "icon": "🔗",
            "desc": "Watershed-based separation of touching and overlapping objects. Prevents under-counting in dense samples.",
            "module": "🔗  Object Separation",
        },
        {
            "num": "04",
            "title": "Focus Quality",
            "icon": "🎯",
            "desc": "Detect and flag out-of-focus regions. Laplacian variance and gradient-based metrics identify blur artefacts that corrupt size measurements.",
            "module": "🎯  Focus & Depth",
        },
        {
            "num": "05",
            "title": "Sampling & Edge Bias",
            "icon": "📊",
            "desc": "Guard frame method and Abercrombie correction to handle edge-touching objects and produce statistically unbiased counts and size estimates.",
            "module": "📊  Sampling & Edge Bias",
        },
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
    st.markdown("### Quick start")
    st.markdown("""
    1. Pick an **Analysis Module** from the sidebar  
    2. **Upload** a microscopy image (TIFF, PNG, JPG)  
    3. Adjust parameters using the interactive controls  
    4. **Download** results as CSV or annotated image  
    """)

    st.info("💡 Each module works independently — you don't need to run them in order.", icon="💡")
