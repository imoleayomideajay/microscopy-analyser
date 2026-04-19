import streamlit as st
import numpy as np
from utils.image_io import load_image
from utils.sampling import (
    segment_objects,
    apply_guard_frame,
    abercrombie_correction,
    measure_all_objects,
    visualise_guard_frame,
    sampling_report,
)

def render():
    st.markdown("## 📊 Sampling & Edge Bias Correction")
    st.markdown("<p style='color:#6868a0;'>Guard frame method and Abercrombie correction for statistically unbiased counting and sizing.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Segmentation")
        thresh_method = st.selectbox("Threshold method", ["Otsu", "Li", "Triangle", "Yen"])
        sigma = st.slider("Pre-blur sigma", 0.0, 5.0, 1.5, step=0.1)
        min_size = st.slider("Min object size (px²)", 10, 1000, 80)
        use_watershed = st.toggle("Watershed separation", value=True)

        st.markdown("### Guard frame (unbiased counting)")
        use_guard = st.toggle("Apply guard frame", value=True)
        if use_guard:
            guard_px = st.slider("Guard frame width (px)", 1, 100, 20,
                                 help="Objects touching the inclusion lines are counted; those touching exclusion lines are not.")
            exclusion_edges = st.multiselect(
                "Exclusion edges",
                ["Top", "Left", "Bottom", "Right"],
                default=["Top", "Left"],
                help="Objects touching these edges are excluded. Standard: exclude top and left."
            )
        else:
            guard_px = 0
            exclusion_edges = []

        st.markdown("### Abercrombie correction")
        use_abercrombie = st.toggle("Apply Abercrombie correction", value=False,
                                    help="For thick sections: corrects for objects that span multiple focal planes.")
        if use_abercrombie:
            section_thickness = st.number_input("Section thickness (µm)", min_value=0.1, value=10.0)
            object_height = st.number_input("Mean object height (µm)", min_value=0.1, value=8.0)

        st.markdown("### Calibration")
        px_size = st.number_input("Pixel size (µm/px)", min_value=0.001, value=0.1, step=0.001, format="%.4f")

    with col_upload:
        uploaded = st.file_uploader("Upload microscopy image", type=["tif","tiff","png","jpg","jpeg"], key="samp_upload")

        if uploaded:
            img_gray, img_rgb = load_image(uploaded)
            h, w = img_gray.shape

            binary, labels_raw, n_raw = segment_objects(img_gray, thresh_method, sigma, min_size, use_watershed)

            if use_guard:
                labels_counted, n_counted, edge_mask = apply_guard_frame(
                    labels_raw, h, w, guard_px, exclusion_edges
                )
            else:
                labels_counted = labels_raw
                n_counted = n_raw
                edge_mask = None

            if use_abercrombie:
                n_corrected = abercrombie_correction(n_counted, section_thickness, object_height, h * px_size, w * px_size)
            else:
                n_corrected = n_counted

            vis = visualise_guard_frame(img_rgb.copy(), labels_counted, labels_raw,
                                         guard_px, exclusion_edges if use_guard else [])
            measurements = measure_all_objects(labels_counted, px_size)
            report = sampling_report(n_raw, n_counted, n_corrected, measurements, h * px_size, w * px_size, px_size)

            tab_vis, tab_data, tab_report = st.tabs(["Guard frame visualisation", "Per-object data", "Sampling report"])

            with tab_vis:
                st.image(vis, use_container_width=True)
                st.caption("🟢 Counted objects  |  🔴 Excluded (touch exclusion edge)  |  🔵 Guard frame boundary")

            with tab_data:
                if measurements:
                    import pandas as pd
                    df = pd.DataFrame(measurements)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                                       file_name="sampling_corrected_measurements.csv", mime="text/csv")
                else:
                    st.info("No counted objects found.")

            with tab_report:
                st.markdown(f"""
                <div class='problem-card'>
                <h4>📊 Counting report</h4>
                <p>
                Raw count (all connected components): <b style='color:#a0a8ff'>{n_raw}</b><br>
                After guard frame exclusion: <b style='color:#a0a8ff'>{n_counted}</b><br>
                {'After Abercrombie correction: <b style="color:#a0a8ff">' + f'{n_corrected:.1f}' + '</b><br>' if use_abercrombie else ''}
                Field area: <b style='color:#a0a8ff'>{report['field_area_um2']:.1f} µm²</b><br>
                Number density: <b style='color:#a0a8ff'>{report['density']:.4f} objects/µm²</b>
                </p>
                </div>
                """, unsafe_allow_html=True)

                if measurements:
                    import pandas as pd
                    df = pd.DataFrame(measurements)
                    st.markdown("**Size distribution summary**")
                    st.dataframe(df[["area_um2", "equiv_diameter_um", "circularity"]].describe().round(3), use_container_width=True)

        else:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35; border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2rem;'>📊</div>
                <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to apply bias correction</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ Edge bias and the guard frame method"):
        st.markdown("""
        **The problem**: Objects that intersect the image border are only partially visible. Including them introduces a negative size bias; excluding them all introduces a negative count bias.

        **Guard frame method** (Howard & Reed, 1998):
        - Draw an inner "counting frame" inset by a guard distance from the image edge  
        - **Include** objects that touch or are inside the inclusion lines (bottom and right)  
        - **Exclude** objects that touch the exclusion lines (top and left)  
        - This gives each object an equal probability of being counted regardless of size

        **Abercrombie correction** (for thick sections):
        > N_true = N_counted × T / (T + h)

        where T = section thickness and h = mean object height. Corrects for objects that span the full section thickness and are thus counted in multiple sections.

        **Field selection bias**: Always randomise or systematically sample fields — never cherry-pick "representative" or "nice-looking" ones. Use a random number generator or a stage encoder to define field positions before viewing.
        """)
