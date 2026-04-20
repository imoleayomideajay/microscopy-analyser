import streamlit as st
import numpy as np
from PIL import Image
import io
from utils.image_io import load_image, make_download_bytes
from utils.cache import cached_measure_calibrated, cached_detect_scale_bar

def render():
    st.markdown("## 📐 Scale Calibration")
    st.markdown("<p style='color:#6868a0;'>Map pixel coordinates to physical units accurately — the foundation of all size measurements.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Calibration settings")
        method = st.selectbox(
            "Calibration method",
            ["Manual (known pixel size)", "Scale bar detection", "Known object size"],
        )

        if method == "Manual (known pixel size)":
            px_size = st.number_input("Pixel size (µm/pixel)", min_value=0.001, max_value=100.0, value=0.1, step=0.001, format="%.4f")
            unit = st.selectbox("Unit", ["µm", "nm", "mm"])
        elif method == "Scale bar detection":
            st.info("Upload an image with a visible scale bar. The tool will detect it automatically.")
            px_size = None
            unit = st.selectbox("Scale bar unit", ["µm", "nm", "mm"])
            known_length = st.number_input("Known scale bar length", min_value=0.1, value=10.0, step=0.1)
        else:
            ref_px = st.number_input("Reference object size in pixels", min_value=1.0, value=50.0)
            ref_um = st.number_input("True size of reference object (µm)", min_value=0.1, value=5.0)
            px_size = ref_um / ref_px
            unit = "µm"
            st.markdown(f"<div class='metric-chip'>Derived: {px_size:.4f} µm/px</div>", unsafe_allow_html=True)

        st.markdown("### Object detection")
        min_area_px = st.slider("Min object area (px²)", 10, 2000, 100)
        max_area_px = st.slider("Max object area (px²)", 100, 50000, 10000)

    with col_upload:
        uploaded = st.file_uploader("Upload microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="scale_upload")

        if uploaded:
            file_bytes = uploaded.read()

            if method == "Scale bar detection":
                px_size, bar_rect = cached_detect_scale_bar(file_bytes, known_length)
                if px_size:
                    st.success(f"Scale bar detected → **{px_size:.4f} {unit}/pixel**")
                else:
                    st.warning("Scale bar not detected — falling back to 0.1 µm/pixel. Set manually.")
                    px_size = 0.1

            with st.spinner("Measuring objects…"):
                overlay, measurements = cached_measure_calibrated(
                    file_bytes, px_size, unit, min_area_px, max_area_px
                )

            tab_img, tab_data = st.tabs(["Annotated image", "Measurements"])
            with tab_img:
                st.image(overlay, use_container_width=True)
            with tab_data:
                if measurements:
                    import pandas as pd
                    df = pd.DataFrame(measurements)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "⬇ Download CSV",
                        df.to_csv(index=False).encode(),
                        file_name="calibrated_measurements.csv",
                        mime="text/csv",
                    )
                    st.markdown(f"""
                    <div class='metric-chip'>Objects: {len(df)}</div>
                    <div class='metric-chip'>Mean area: {df['area_'+unit+'2'].mean():.2f} {unit}²</div>
                    <div class='metric-chip'>Mean diameter: {df['equiv_diameter_'+unit].mean():.2f} {unit}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No objects found in the specified area range.")
        else:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35; border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2rem;'>📐</div>
                <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to begin calibration</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ How scale calibration works"):
        st.markdown("""
        **Why it matters**: Every area, length, and diameter measurement depends on knowing how many micrometres a single pixel represents. An error of 10% in pixel size compounds to a 21% error in area.

        **Methods compared**:
        | Method | Best for | Limitation |
        |--------|----------|-----------|
        | Manual (µm/px) | Known microscope specs | Requires metadata |
        | Scale bar detection | Images with embedded bars | Bar must be high-contrast |
        | Known object | Calibration beads | Need a reference object |

        **Camera binning**: If you bin 2×2 on your camera, your effective pixel size doubles. Always check binning settings in your acquisition software.
        """)
