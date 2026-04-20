import streamlit as st
import numpy as np
from utils.image_io import load_image
from utils.separation import (
    segment_binary,
    apply_watershed,
    overlay_labels,
    compare_counts,
)

def render():
    st.markdown("## 🔗 Object Separation")
    st.markdown("<p style='color:#6868a0;'>Watershed-based splitting of touching and overlapping objects to prevent under-counting.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Segmentation")
        thresh_method = st.selectbox("Threshold method", ["Otsu", "Li", "Triangle", "Yen"])
        sigma_pre = st.slider("Pre-blur sigma", 0.0, 5.0, 1.5, step=0.1)
        min_size = st.slider("Min object size (px²)", 10, 1000, 80)

        st.markdown("### Watershed settings")
        use_watershed = st.toggle("Apply watershed separation", value=True)
        if use_watershed:
            min_distance = st.slider("Min peak distance (px)", 3, 50, 10,
                                     help="Minimum distance between object centres. Increase for larger objects.")
            compactness = st.slider("Compactness", 0.0, 1.0, 0.0, step=0.05,
                                    help="0 = standard watershed; higher values produce rounder segments.")
            footprint_size = st.slider("Marker footprint (px)", 3, 25, 7, step=2)
        else:
            min_distance, compactness, footprint_size = 10, 0.0, 7

        st.markdown("### Visualisation")
        show_markers = st.toggle("Show seed markers", value=True)
        alpha = st.slider("Overlay opacity", 0.1, 1.0, 0.45, step=0.05)

    with col_upload:
        uploaded = st.file_uploader("Upload microscopy image", type=["tif","tiff","png","jpg","jpeg"], key="sep_upload")

        if uploaded:
            img_gray, img_rgb = load_image(uploaded)

            binary = segment_binary(img_gray, thresh_method, sigma_pre, min_size)

            if use_watershed:
                labels_ws, n_ws, markers = apply_watershed(
                    binary, img_gray, min_distance, compactness, footprint_size
                )
            else:
                from skimage import measure
                labels_ws = measure.label(binary)
                n_ws = labels_ws.max()
                markers = None

            from skimage import measure
            labels_naive = measure.label(binary)
            n_naive = labels_naive.max()

            overlay_naive = overlay_labels(img_rgb.copy(), labels_naive, alpha)
            overlay_ws = overlay_labels(img_rgb.copy(), labels_ws, alpha,
                                        markers=markers if show_markers else None)

            tab_naive, tab_ws, tab_compare = st.tabs(["Naive (no separation)", "After watershed", "Comparison"])
            with tab_naive:
                st.image(overlay_naive, use_container_width=True)
                st.metric("Objects (naive)", n_naive)
            with tab_ws:
                st.image(overlay_ws, use_container_width=True)
                st.metric("Objects (watershed)", n_ws)
            with tab_compare:
                diff = n_ws - n_naive
                st.metric("Extra objects found by watershed", diff, delta=f"+{diff}" if diff > 0 else str(diff))
                stats = compare_counts(labels_naive, labels_ws)
                import pandas as pd
                st.dataframe(pd.DataFrame(stats), use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35; border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2rem;'>🔗</div>
                <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to begin separation analysis</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ How watershed separation works"):
        st.markdown("""
        **The problem**: Standard connected-component labelling treats any touching group of pixels as a single object. In a dense cell culture or particle suspension this massively under-counts.

        **Watershed** treats the distance-transform of the binary mask as a topographic surface. It "floods" from seed points (local maxima of the distance transform) and places boundaries where flooding fronts meet — exactly at the narrowest connections between touching objects.

        **Parameters to tune**:
        - **Min peak distance** — increase if objects are large; decrease for small, dense objects  
        - **Compactness** — higher values regularise boundaries, useful for round cells  
        - **Footprint** — controls the neighbourhood used to find local maxima

        **Limitations**: Watershed struggles when objects overlap heavily (not just touch) or when shape is highly irregular. In those cases, deep-learning segmentation (e.g. Cellpose) is recommended.
        """)
