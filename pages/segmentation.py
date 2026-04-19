import streamlit as st
import numpy as np
from utils.image_io import load_image
from utils.segmentation import (
    correct_background,
    apply_threshold,
    get_threshold_methods,
    segment_and_label,
    overlay_segmentation,
    compute_segmentation_stats,
)

def render():
    st.markdown("## ✂️ Image Segmentation")
    st.markdown("<p style='color:#6868a0;'>Robust object/background separation using adaptive methods and background correction.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Pre-processing")
        bg_correct = st.toggle("Background correction (rolling ball)", value=True)
        if bg_correct:
            ball_radius = st.slider("Rolling ball radius (px)", 10, 300, 50)
        else:
            ball_radius = 50

        denoise = st.toggle("Denoise (Gaussian blur)", value=True)
        if denoise:
            sigma = st.slider("Blur sigma", 0.5, 5.0, 1.0, step=0.1)
        else:
            sigma = 0.0

        st.markdown("### Thresholding")
        method = st.selectbox("Method", get_threshold_methods())
        if method == "Manual":
            manual_thresh = st.slider("Threshold value", 0, 255, 128)
        else:
            manual_thresh = None

        invert = st.toggle("Invert (dark objects on bright background)", value=False)

        st.markdown("### Post-processing")
        min_size = st.slider("Remove objects smaller than (px²)", 0, 500, 50)
        fill_holes = st.toggle("Fill holes", value=True)

    with col_upload:
        uploaded = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="seg_upload")

        if uploaded:
            img_gray, img_rgb = load_image(uploaded)

            # Run pipeline
            corrected = correct_background(img_gray, ball_radius) if bg_correct else img_gray
            binary, thresh_val = apply_threshold(corrected, method, sigma, manual_thresh, invert)
            labeled, n_objects = segment_and_label(binary, min_size, fill_holes)
            overlay = overlay_segmentation(img_rgb, labeled, n_objects)
            stats = compute_segmentation_stats(labeled, n_objects)

            tab_orig, tab_binary, tab_overlay = st.tabs(["Original", "Binary mask", "Labelled overlay"])
            with tab_orig:
                st.image(img_rgb, use_container_width=True)
            with tab_binary:
                st.image((binary * 255).astype(np.uint8), use_container_width=True)
            with tab_overlay:
                st.image(overlay, use_container_width=True)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Objects found", n_objects)
            c2.metric("Threshold value", thresh_val if thresh_val else method)
            c3.metric("Coverage (%)", f"{stats['coverage_pct']:.1f}")
            c4.metric("Mean object size (px²)", f"{stats['mean_area']:.0f}")

        else:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35; border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2rem;'>✂️</div>
                <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to begin segmentation</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ Segmentation methods explained"):
        st.markdown("""
        | Method | Description | Best for |
        |--------|-------------|---------|
        | **Otsu** | Minimises intra-class variance | Bimodal histograms |
        | **Li** | Minimises cross-entropy | Cells with variable intensity |
        | **Triangle** | Works on skewed histograms | Fluorescence (sparse signal) |
        | **Yen** | Maximises image entropy | High-contrast images |
        | **Sauvola** | Local adaptive | Uneven illumination |
        | **Niblack** | Mean–std local | Textured backgrounds |
        | **Manual** | User-defined | Full control |

        **Rolling ball background correction** estimates and subtracts a smoothly varying background intensity, greatly improving segmentation quality on images with uneven illumination (common in widefield fluorescence and brightfield).
        """)
