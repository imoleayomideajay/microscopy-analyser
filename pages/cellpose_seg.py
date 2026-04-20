import streamlit as st
import numpy as np
import cv2
from skimage import measure


def render():
    st.markdown("## 🧬 Cellpose Segmentation")
    st.markdown(
        "<p style='color:#6868a0;'>Deep-learning cell segmentation using Cellpose — outperforms classical "
        "methods on complex morphologies, touching cells, and variable illumination.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Cellpose availability check ───────────────────────
    try:
        from cellpose import models as cp_models
        cellpose_available = True
    except ImportError:
        cellpose_available = False

    if not cellpose_available:
        st.warning(
            "**Cellpose is not installed** in this environment. "
            "Install it with `pip install cellpose` and restart the app.",
            icon="⚠️",
        )
        with st.expander("Installation instructions"):
            st.code(
                "pip install cellpose\n"
                "# GPU support (optional, requires CUDA):\n"
                "pip install cellpose[gui]",
                language="bash",
            )
            st.markdown("""
            Cellpose is a generalist segmentation model trained on ~700 000 manually annotated cells.
            It handles brightfield, phase contrast, and fluorescence images without retraining.

            **Models available**:
            | Model | Best for |
            |-------|---------|
            | `cyto3` | Cytoplasm / general cells (default) |
            | `nuclei` | DAPI / Hoechst nuclear stains |
            | `cyto2` | Cytoplasm, older generation |
            | `tissuenet` | Tissue sections |
            | `livecell` | Phase contrast live cells |
            """)
        return

    col_upload, col_params = st.columns([1.6, 1])

    with col_params:
        st.markdown("### Model")
        model_type = st.selectbox(
            "Cellpose model",
            ["cyto3", "nuclei", "cyto2", "tissuenet", "livecell"],
            help="cyto3 is the best general-purpose model.",
        )

        st.markdown("### Channels")
        st.caption(
            "Tell Cellpose which channel contains the signal. "
            "0 = grayscale. For fluorescence: set channel 1 to the cell channel."
        )
        chan1 = st.selectbox(
            "Cell channel (1=red, 2=green, 3=blue, 0=gray)",
            [0, 1, 2, 3],
            index=0,
        )
        chan2 = st.selectbox(
            "Nuclear channel (0 = none)",
            [0, 1, 2, 3],
            index=0,
        )

        st.markdown("### Segmentation parameters")
        diameter = st.slider(
            "Expected cell diameter (px, 0 = auto-estimate)",
            0, 300, 0, step=5,
            help="Average diameter of your cells in pixels. Set 0 to let Cellpose estimate it.",
        )
        flow_threshold = st.slider(
            "Flow threshold",
            0.0, 3.0, 0.4, step=0.05,
            help="Higher = more permissive (more cells found, more false positives).",
        )
        cellprob_threshold = st.slider(
            "Cell probability threshold",
            -6.0, 6.0, 0.0, step=0.5,
            help="Lower = more cells included. Increase to reduce false positives.",
        )

        st.markdown("### Measurements")
        px_size = st.number_input(
            "Pixel size (µm/px)",
            min_value=0.001, value=0.1, step=0.001, format="%.4f",
        )
        unit = st.selectbox("Unit", ["µm", "nm", "mm"])
        min_size_px = st.slider("Min cell area (px²)", 10, 2000, 100)

    with col_upload:
        uploaded = st.file_uploader(
            "Upload microscopy image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="cellpose_upload",
        )

        if uploaded:
            file_bytes = uploaded.read()

            with st.spinner("Running Cellpose… this may take 15–60 s on CPU"):
                try:
                    from utils.cache import cached_cellpose
                    masks, flows, styles, rgb, gray = cached_cellpose(
                        file_bytes,
                        model_type=model_type,
                        diameter=float(diameter),
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        channels=[chan1, chan2],
                    )
                except Exception as e:
                    st.error(f"Cellpose error: {e}")
                    st.stop()

            # Filter by min size
            props_all = measure.regionprops(masks)
            keep = {p.label for p in props_all if p.area >= min_size_px}
            filtered_masks = masks.copy()
            for p in props_all:
                if p.label not in keep:
                    filtered_masks[masks == p.label] = 0

            n_cells = len(keep)
            props = measure.regionprops(filtered_masks, intensity_image=gray)

            # ── Overlay ──────────────────────────────────
            overlay = _make_overlay(rgb, filtered_masks, n_cells)

            # ── Measurements ─────────────────────────────
            measurements = []
            for p in props:
                if p.label == 0:
                    continue
                area_cal = p.area * (px_size ** 2)
                diam_cal = p.equivalent_diameter_area * px_size
                perim_cal = p.perimeter * px_size
                circ = (4 * np.pi * p.area) / (p.perimeter ** 2 + 1e-9)
                measurements.append({
                    "label": p.label,
                    f"area_{unit}²": round(area_cal, 4),
                    f"equiv_diameter_{unit}": round(diam_cal, 4),
                    f"perimeter_{unit}": round(perim_cal, 4),
                    "circularity": round(float(circ), 4),
                    "eccentricity": round(float(p.eccentricity), 4),
                    "solidity": round(float(p.solidity), 4),
                    "mean_intensity": round(float(p.mean_intensity), 2),
                })

            # ── Tabs ─────────────────────────────────────
            tab_ov, tab_mask, tab_data, tab_dist = st.tabs([
                "Overlay", "Mask", "Measurements", "Distributions"
            ])

            with tab_ov:
                st.image(overlay, use_container_width=True)

            with tab_mask:
                # Colour-coded label image
                label_rgb = _label_to_rgb(filtered_masks)
                st.image(label_rgb, use_container_width=True)

            with tab_data:
                if measurements:
                    import pandas as pd
                    df = pd.DataFrame(measurements)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "⬇ Download CSV",
                        df.to_csv(index=False).encode(),
                        file_name=f"cellpose_{model_type}_measurements.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No cells found after size filtering.")

            with tab_dist:
                _plot_distributions(measurements, unit)

            # ── Summary metrics ───────────────────────────
            st.markdown("---")
            areas = [m[f"area_{unit}²"] for m in measurements]
            diams = [m[f"equiv_diameter_{unit}"] for m in measurements]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Cells found", n_cells)
            c2.metric(f"Mean area ({unit}²)", f"{np.mean(areas):.2f}" if areas else "—")
            c3.metric(f"Median diameter ({unit})", f"{np.median(diams):.2f}" if diams else "—")
            c4.metric("Mean circularity", f"{np.mean([m['circularity'] for m in measurements]):.3f}" if measurements else "—")
            c5.metric("Model", model_type)

        else:
            st.markdown(
                """
                <div style='background:#0f0f1a; border:1.5px dashed #1e1e35;
                     border-radius:12px; padding:2.5rem; text-align:center; color:#3a3a60;'>
                    <div style='font-size:2rem;'>🧬</div>
                    <div style='margin-top:0.5rem; font-size:0.9rem;'>Upload an image to run Cellpose</div>
                    <div style='margin-top:0.3rem; font-size:0.78rem; color:#2a2a50;'>
                        First run may be slow — model weights are downloaded once and cached
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("ℹ️ How Cellpose works"):
        st.markdown("""
        Cellpose uses a **U-Net style neural network** that predicts two vector fields (horizontal and
        vertical "flows") pointing from each pixel toward its cell's centre. Cells are then reconstructed
        by following those flows — a fundamentally different approach from thresholding or watershed.

        **Why it outperforms classical methods**:
        - Handles touching cells without needing watershed tuning
        - Robust to uneven illumination, out-of-focus halos, and staining variability
        - Works across modalities: brightfield, phase contrast, fluorescence

        **Diameter matters**: If you know the approximate cell diameter in pixels, set it — the model
        internally rescales the image to that diameter before inference. Setting it to 0 triggers
        automatic estimation (adds ~5 s to run time).

        **GPU acceleration**: Cellpose runs on CPU by default here. On a local machine with a CUDA GPU,
        inference is 10–50× faster. Install `torch` with CUDA and Cellpose will detect it automatically.

        **Reference**: Stringer et al., *Nature Methods* 2021.
        """)


# ─── helpers ─────────────────────────────────────────────

def _make_overlay(rgb: np.ndarray, masks: np.ndarray, n: int, alpha: float = 0.40):
    np.random.seed(42)
    colors = np.random.randint(60, 230, size=(n + 2, 3), dtype=np.uint8)
    color_img = np.zeros_like(rgb)
    for lbl in range(1, int(masks.max()) + 1):
        color_img[masks == lbl] = colors[lbl % (n + 2)]
    out = cv2.addWeighted(rgb, 1 - alpha, color_img, alpha, 0)
    for lbl in range(1, int(masks.max()) + 1):
        m = (masks == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in colors[lbl % (n + 2)]), 1)
    return out


def _label_to_rgb(masks: np.ndarray):
    from skimage.color import label2rgb
    rgb = (label2rgb(masks, bg_label=0) * 255).astype(np.uint8)
    return rgb


def _plot_distributions(measurements, unit):
    if not measurements:
        st.info("No measurements to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), facecolor="#06060c")
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="#6868a0", labelsize=8)
        ax.spines[:].set_color("#1e1e35")

    areas = [m[f"area_{unit}²"] for m in measurements]
    diams = [m[f"equiv_diameter_{unit}"] for m in measurements]
    circs = [m["circularity"] for m in measurements]

    axes[0].hist(areas, bins=30, color="#a0a8ff", alpha=0.85, edgecolor="#06060c")
    axes[0].set_xlabel(f"Area ({unit}²)", color="#6868a0", fontsize=8)
    axes[0].set_ylabel("Count", color="#6868a0", fontsize=8)
    axes[0].set_title("Area distribution", color="#c8c8d8", fontsize=9)

    axes[1].hist(diams, bins=30, color="#60d0a8", alpha=0.85, edgecolor="#06060c")
    axes[1].set_xlabel(f"Diameter ({unit})", color="#6868a0", fontsize=8)
    axes[1].set_title("Diameter distribution", color="#c8c8d8", fontsize=9)

    axes[2].hist(circs, bins=30, color="#ffb060", alpha=0.85, edgecolor="#06060c")
    axes[2].set_xlabel("Circularity (0–1)", color="#6868a0", fontsize=8)
    axes[2].set_title("Circularity distribution", color="#c8c8d8", fontsize=9)

    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)
