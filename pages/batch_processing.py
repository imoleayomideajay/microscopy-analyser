import streamlit as st
import numpy as np
import pandas as pd
import zipfile
import io
import time
from utils.cache import cached_batch_single


def render():
    st.markdown("## 🗂️ Batch Processing")
    st.markdown(
        "<p style='color:#6868a0;'>Upload multiple images and run the full analysis pipeline "
        "on all of them. Results export as a single consolidated CSV.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_params, col_main = st.columns([1, 2])

    with col_params:
        st.markdown("### Pipeline settings")

        with st.expander("📐 Scale", expanded=True):
            px_size = st.number_input("Pixel size (µm/px)", min_value=0.001, value=0.1,
                                      step=0.001, format="%.4f", key="b_px")
            unit = st.selectbox("Unit", ["µm", "nm", "mm"], key="b_unit")
            min_area_px = st.slider("Min object area (px²)", 10, 2000, 100, key="b_minarea")
            max_area_px = st.slider("Max object area (px²)", 100, 50000, 10000, key="b_maxarea")

        with st.expander("✂️ Segmentation", expanded=True):
            seg_method = st.selectbox("Threshold method",
                                      ["Otsu", "Li", "Triangle", "Yen"], key="b_seg")
            sigma = st.slider("Pre-blur sigma", 0.0, 5.0, 1.5, step=0.1, key="b_sigma")
            min_size = st.slider("Min object size (px²)", 10, 1000, 80, key="b_minsize")
            use_watershed = st.toggle("Watershed separation", value=True, key="b_ws")

        with st.expander("🎯 Focus quality"):
            focus_metric = st.selectbox("Focus metric", [
                "Laplacian variance",
                "Tenengrad (Sobel gradient energy)",
                "Normalised variance",
                "Brenner gradient",
            ], key="b_focus")
            tile_size = st.slider("Tile size (px)", 16, 256, 64, step=16, key="b_tile")
            blur_percentile = st.slider("Blur threshold (percentile)", 1, 50, 20, key="b_blur")

        with st.expander("📊 Edge bias"):
            guard_px = st.slider("Guard frame width (px)", 0, 100, 20, key="b_guard")
            excl_edges = st.multiselect(
                "Exclusion edges", ["Top", "Left", "Bottom", "Right"],
                default=["Top", "Left"], key="b_excl",
            )

        with st.expander("⚙️ Batch options"):
            flag_blurry = st.toggle("Flag blurry images in output", value=True, key="b_flagblur")
            blur_warn_pct = st.slider("Blurry tile % warning threshold", 5, 80, 30, key="b_warnpct")
            skip_blurry = st.toggle("Skip images above blur threshold", value=False, key="b_skip")

    with col_main:
        uploaded_files = st.file_uploader(
            "Upload images (multiple allowed)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if not uploaded_files:
            st.markdown("""
            <div style='background:#0f0f1a; border:1.5px dashed #1e1e35;
                 border-radius:12px; padding:3rem; text-align:center; color:#3a3a60;'>
                <div style='font-size:2.5rem;'>🗂️</div>
                <div style='margin-top:0.6rem; font-size:0.95rem;'>Upload one or more images to begin</div>
                <div style='margin-top:0.3rem; font-size:0.78rem; color:#2a2a50;'>
                    TIFF · PNG · JPEG &nbsp;·&nbsp; Each image is processed independently
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown(f"**{len(uploaded_files)} image{'s' if len(uploaded_files) > 1 else ''} queued**")

        # Preview strip
        n_preview = min(len(uploaded_files), 6)
        preview_cols = st.columns(n_preview)
        for i, f in enumerate(uploaded_files[:n_preview]):
            from PIL import Image
            img = Image.open(f).convert("RGB")
            img.thumbnail((120, 120))
            preview_cols[i].image(img, caption=f.name[:18], use_container_width=True)
            f.seek(0)

        st.markdown("---")

        run_col, _ = st.columns([1, 3])
        with run_col:
            run = st.button("▶  Run batch analysis", use_container_width=True)

        if run:
            excl_str = ",".join(excl_edges)
            results = []
            errors = []

            progress = st.progress(0, text="Starting…")
            status_box = st.empty()
            log_lines = []

            for i, f in enumerate(uploaded_files):
                fname = f.name
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing {i+1}/{len(uploaded_files)}: {fname}",
                )
                status_box.markdown(
                    f"<div style='font-family:DM Mono,monospace; font-size:0.78rem; "
                    f"color:#6868a0;'>{'<br>'.join(log_lines[-6:])}</div>",
                    unsafe_allow_html=True,
                )

                try:
                    t0 = time.perf_counter()
                    file_bytes = f.read()

                    row = cached_batch_single(
                        file_bytes=file_bytes,
                        filename=fname,
                        seg_method=seg_method,
                        sigma=sigma,
                        min_size=min_size,
                        use_watershed=use_watershed,
                        px_size=px_size,
                        unit=unit,
                        min_area_px=min_area_px,
                        max_area_px=max_area_px,
                        focus_metric=focus_metric,
                        tile_size=tile_size,
                        blur_percentile=blur_percentile,
                        guard_px=guard_px,
                        exclusion_edges_str=excl_str,
                    )

                    elapsed = time.perf_counter() - t0
                    is_blurry = row["pct_blurry_tiles"] > blur_warn_pct

                    if skip_blurry and is_blurry:
                        log_lines.append(f"  ⚠ SKIPPED (blurry): {fname}")
                        continue

                    if flag_blurry:
                        row["blurry_flag"] = "⚠ BLURRY" if is_blurry else "OK"

                    row["processing_s"] = round(elapsed, 2)
                    results.append(row)
                    log_lines.append(f"  ✓ {fname} — {row['n_counted']} objects ({elapsed:.1f}s)")

                except Exception as e:
                    errors.append({"filename": fname, "error": str(e)})
                    log_lines.append(f"  ✗ ERROR {fname}: {e}")

            progress.progress(1.0, text="Complete ✓")
            status_box.empty()

            if not results:
                st.error("No images processed successfully.")
                if errors:
                    st.dataframe(pd.DataFrame(errors), use_container_width=True)
                return

            df = pd.DataFrame(results)

            # ── Results ──────────────────────────────────
            st.markdown("### Results")

            # Summary bar
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Images processed", len(df))
            c2.metric("Images skipped", len(uploaded_files) - len(df) - len(errors))
            c3.metric("Errors", len(errors))
            total_cells = int(df["n_counted"].sum())
            c4.metric("Total objects counted", total_cells)
            mean_blur = df["pct_blurry_tiles"].mean()
            c5.metric("Mean blur %", f"{mean_blur:.1f}")

            tab_table, tab_plots, tab_errors = st.tabs(["Data table", "Summary plots", "Errors"])

            with tab_table:
                # Highlight blurry rows
                def _style_row(row):
                    if flag_blurry and row.get("blurry_flag") == "⚠ BLURRY":
                        return ["background-color: #1a0a0a"] * len(row)
                    return [""] * len(row)

                styled = df.style.apply(_style_row, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)

                # Downloads
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇ Download results CSV",
                        df.to_csv(index=False).encode(),
                        file_name="batch_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl2:
                    # Bundle CSV + per-image JSON metadata in a zip
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("batch_results.csv", df.to_csv(index=False))
                        import json
                        params = {
                            "seg_method": seg_method, "sigma": sigma,
                            "min_size": min_size, "use_watershed": use_watershed,
                            "px_size": px_size, "unit": unit,
                            "focus_metric": focus_metric, "tile_size": tile_size,
                            "blur_percentile": blur_percentile,
                            "guard_px": guard_px, "exclusion_edges": excl_edges,
                        }
                        zf.writestr("pipeline_params.json", json.dumps(params, indent=2))
                    st.download_button(
                        "⬇ Download ZIP (CSV + params)",
                        zip_buf.getvalue(),
                        file_name="batch_results.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            with tab_plots:
                _render_summary_plots(df, unit)

            with tab_errors:
                if errors:
                    st.dataframe(pd.DataFrame(errors), use_container_width=True)
                else:
                    st.success("No errors — all images processed successfully.")


# ─── plotting ─────────────────────────────────────────────

def _render_summary_plots(df: pd.DataFrame, unit: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area_col = f"mean_area_{unit}2"
    diam_col = f"mean_diameter_{unit}"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#06060c")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="#6868a0", labelsize=8)
        ax.spines[:].set_color("#1e1e35")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # 1. Objects per image
    axes[0].bar(range(len(df)), df["n_counted"], color="#a0a8ff", alpha=0.85, width=0.7)
    axes[0].set_title("Objects counted per image", color="#c8c8d8", fontsize=9)
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(
        [n[:14] for n in df["filename"]],
        rotation=45, ha="right", fontsize=7, color="#6868a0",
    )
    axes[0].set_ylabel("Count", color="#6868a0", fontsize=8)

    # 2. Mean area distribution across images
    if area_col in df.columns:
        axes[1].bar(range(len(df)), df[area_col], color="#60d0a8", alpha=0.85, width=0.7)
        axes[1].set_title(f"Mean object area ({unit}²) per image", color="#c8c8d8", fontsize=9)
        axes[1].set_xticks(range(len(df)))
        axes[1].set_xticklabels(
            [n[:14] for n in df["filename"]],
            rotation=45, ha="right", fontsize=7, color="#6868a0",
        )
        axes[1].set_ylabel(f"Area ({unit}²)", color="#6868a0", fontsize=8)

    # 3. Blur % per image
    colors_blur = ["#ff6060" if v > 30 else "#a0a8ff" for v in df["pct_blurry_tiles"]]
    axes[2].bar(range(len(df)), df["pct_blurry_tiles"], color=colors_blur, alpha=0.85, width=0.7)
    axes[2].axhline(30, color="#ff6060", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[2].set_title("Blurry tiles (%) per image", color="#c8c8d8", fontsize=9)
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels(
        [n[:14] for n in df["filename"]],
        rotation=45, ha="right", fontsize=7, color="#6868a0",
    )
    axes[2].set_ylabel("Blurry %", color="#6868a0", fontsize=8)

    # 4. Density scatter (objects/µm² vs mean circularity)
    if "density_per_um2" in df.columns and "mean_circularity" in df.columns:
        sc = axes[3].scatter(
            df["density_per_um2"], df["mean_circularity"],
            c=df["pct_blurry_tiles"], cmap="plasma",
            s=60, alpha=0.85, edgecolors="#0f0f1a", linewidth=0.5,
        )
        axes[3].set_xlabel("Density (objects/µm²)", color="#6868a0", fontsize=8)
        axes[3].set_ylabel("Mean circularity", color="#6868a0", fontsize=8)
        axes[3].set_title("Density vs circularity (colour = blur %)", color="#c8c8d8", fontsize=9)
        cbar = fig.colorbar(sc, ax=axes[3], pad=0.02)
        cbar.ax.tick_params(colors="#6868a0", labelsize=7)
        cbar.set_label("Blur %", color="#6868a0", fontsize=7)

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close(fig)
