import streamlit as st
import numpy as np
import io

def render():
    st.markdown("## 🎯 Focus Quality Assessment")
    st.markdown("<p style='color:#6868a0;'>Detect out-of-focus regions that corrupt size and shape measurements.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Focus metric")
        metric = st.selectbox("Sharpness estimator", [
            "Laplacian variance",
            "Normalised variance",
            "Tenengrad (Sobel gradient energy)",
            "Brenner gradient",
        ])

        st.markdown("### Local analysis")
        tile_size = st.slider("Tile size (px)", 16, 256, 64, step=16,
                              help="Image is divided into tiles; focus is computed per tile.")
        blur_threshold = st.slider(
            "Blur threshold (percentile)",
            1, 50, 20,
            help="Tiles below this percentile of sharpness are flagged as blurry."
        )

        st.markdown("### Visualisation")
        cmap = st.selectbox("Heatmap colourmap", ["plasma", "inferno", "viridis", "magma", "hot"])
        show_grid = st.toggle("Show tile grid", value=True)
        show_flagged = st.toggle("Highlight blurry tiles", value=True)

    with col_upload:
        uploaded = st.file_uploader("Upload microscopy image", type=["tif","tiff","png","jpg","jpeg"], key="focus_upload")

        if uploaded:
            file_bytes = uploaded.read()
            from utils.cache import cached_focus_map
            from utils.focus import flag_blurry_regions, annotate_focus_map

            with st.spinner("Computing focus map…"):
                focus_map, tile_scores, g_metrics, img_rgb = cached_focus_map(
                    file_bytes, metric, tile_size
                )

            blurry_mask = flag_blurry_regions(tile_scores, blur_threshold)
            annotated = annotate_focus_map(img_rgb.copy(), focus_map, blurry_mask,
                                           tile_size, cmap, show_grid, show_flagged)

            tab_heatmap, tab_orig, tab_hist = st.tabs(["Focus heatmap", "Original", "Score distribution"])

            with tab_heatmap:
                st.image(annotated, use_container_width=True)

            with tab_orig:
                st.image(img_rgb, use_container_width=True)

            with tab_hist:
                import pandas as pd
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 3), facecolor="#06060c")
                ax.set_facecolor("#0f0f1a")
                flat = tile_scores.flatten()
                thresh_val = np.percentile(flat, blur_threshold)
                ax.hist(flat, bins=40, color="#a0a8ff", alpha=0.8, edgecolor="#0f0f1a")
                ax.axvline(thresh_val, color="#ff6060", linewidth=1.5, linestyle="--", label=f"Blur threshold ({blur_threshold}th pct)")
                ax.set_xlabel("Sharpness score", color="#6868a0")
                ax.set_ylabel("Tile count", color="#6868a0")
                ax.tick_params(colors="#6868a0")
                ax.spines[:].set_color("#1e1e35")
                ax.legend(facecolor="#0f0f1a", labelcolor="#a0a8ff", edgecolor="#1e1e35", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            pct_blurry = 100 * blurry_mask.sum() / blurry_mask.size
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Global Laplacian var.", f"{g_metrics['laplacian_var']:.1f}")
            c2.metric("Global Tenengrad", f"{g_metrics['tenengrad']:.1f}")
            c3.metric("Blurry tiles (%)", f"{pct_blurry:.1f}")
            c4.metric("Focus verdict", "⚠️ Blurry" if pct_blurry > 30 else "✅ Sharp")

            if pct_blurry > 30:
                st.warning(f"**{pct_blurry:.0f}% of tiles are flagged as blurry.** Measurements from these regions will have inflated area values and distorted shape descriptors. Consider re-acquiring this field or restricting analysis to the sharp tiles.")

        else:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35; border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2rem;'>🎯</div>
                <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to assess focus quality</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ Focus metrics explained"):
        st.markdown("""
        | Metric | Formula | Strengths |
        |--------|---------|-----------|
        | **Laplacian variance** | var(∇²I) | Fast, sensitive to fine detail |
        | **Normalised variance** | var(I)/mean(I) | Illumination-independent |
        | **Tenengrad** | Σ(Gx²+Gy²) where G>T | Robust to noise |
        | **Brenner** | Σ(I(x+2)−I(x))² | Simple, directional |

        **Rule of thumb**: Laplacian variance < 100 on a normalised 8-bit image usually indicates significant blur. The threshold depends strongly on the type of sample and microscopy modality — calibrate on known-good images from your own system.

        **2D vs 3D**: In a z-stack, run this tool on each slice to find the in-focus plane before maximum-intensity projection. In a single 2D image, blurry regions often correspond to sample that is outside the focal plane — caused by non-flat substrates or thick samples.
        """)
