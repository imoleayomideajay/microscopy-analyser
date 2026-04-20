"""
Centralised caching layer.

All heavy image-processing functions are wrapped here with @st.cache_data so
that slider adjustments in the UI don't re-run the full pipeline from scratch.

Each function accepts only hashable primitives (bytes, int, float, str, bool)
so Streamlit can build a reliable cache key.
"""

import numpy as np
import streamlit as st
from PIL import Image
import io


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _bytes_to_arrays(file_bytes: bytes):
    """bytes → (gray uint8, rgb uint8)."""
    pil = Image.open(io.BytesIO(file_bytes))
    try:
        pil.seek(0)
    except Exception:
        pass
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    rgb = np.array(pil.convert("RGB"), dtype=np.uint8)
    gray = np.array(pil.convert("L"), dtype=np.uint8)
    return gray, rgb


# ─────────────────────────────────────────────
# Problem 1 – Scale calibration
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_measure_calibrated(file_bytes: bytes, px_size: float, unit: str,
                               min_area_px: int, max_area_px: int):
    from utils.scale import measure_objects_calibrated, draw_scale_overlay
    gray, rgb = _bytes_to_arrays(file_bytes)
    overlay, measurements = measure_objects_calibrated(
        gray, rgb.copy(), px_size, unit, min_area_px, max_area_px
    )
    overlay = draw_scale_overlay(overlay, px_size, unit)
    return overlay, measurements


@st.cache_data(show_spinner=False)
def cached_detect_scale_bar(file_bytes: bytes, known_length: float):
    from utils.scale import detect_scale_bar
    gray, _ = _bytes_to_arrays(file_bytes)
    return detect_scale_bar(gray, known_length)


# ─────────────────────────────────────────────
# Problem 2 – Segmentation
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_background_correct(file_bytes: bytes, ball_radius: int):
    from utils.segmentation import correct_background
    gray, _ = _bytes_to_arrays(file_bytes)
    return correct_background(gray, ball_radius)


@st.cache_data(show_spinner=False)
def cached_segment(file_bytes: bytes, method: str, sigma: float,
                   manual_thresh, invert: bool,
                   ball_radius: int, bg_correct: bool,
                   min_size: int, fill_holes: bool):
    from utils.segmentation import (
        correct_background, apply_threshold,
        segment_and_label, overlay_segmentation,
        compute_segmentation_stats,
    )
    gray, rgb = _bytes_to_arrays(file_bytes)
    corrected = correct_background(gray, ball_radius) if bg_correct else gray
    binary, thresh_val = apply_threshold(corrected, method, sigma, manual_thresh, invert)
    labeled, n_objects = segment_and_label(binary, min_size, fill_holes)
    overlay = overlay_segmentation(rgb, labeled, n_objects)
    stats = compute_segmentation_stats(labeled, n_objects)
    return binary, labeled, n_objects, overlay, stats, thresh_val


# ─────────────────────────────────────────────
# Problem 3 – Object separation
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_watershed(file_bytes: bytes, thresh_method: str, sigma_pre: float,
                     min_size: int, min_distance: int,
                     compactness: float, footprint_size: int):
    from utils.separation import segment_binary, apply_watershed, overlay_labels
    from skimage import measure
    gray, rgb = _bytes_to_arrays(file_bytes)
    binary = segment_binary(gray, thresh_method, sigma_pre, min_size)
    labels_naive = measure.label(binary)
    n_naive = int(labels_naive.max())
    labels_ws, n_ws, markers = apply_watershed(
        binary, gray, min_distance, compactness, footprint_size
    )
    return binary, labels_naive, n_naive, labels_ws, n_ws, markers, rgb


# ─────────────────────────────────────────────
# Problem 4 – Focus quality
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_focus_map(file_bytes: bytes, metric_name: str, tile_size: int):
    from utils.focus import compute_focus_map, global_focus_metrics
    gray, rgb = _bytes_to_arrays(file_bytes)
    focus_map, tile_scores = compute_focus_map(gray, metric_name, tile_size)
    g_metrics = global_focus_metrics(gray)
    return focus_map, tile_scores, g_metrics, rgb


# ─────────────────────────────────────────────
# Problem 5 – Sampling & edge bias
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_segment_objects(file_bytes: bytes, thresh_method: str,
                            sigma: float, min_size: int, use_watershed: bool):
    from utils.sampling import segment_objects
    gray, rgb = _bytes_to_arrays(file_bytes)
    binary, labels_raw, n_raw = segment_objects(
        gray, thresh_method, sigma, min_size, use_watershed
    )
    return binary, labels_raw, n_raw, rgb


# ─────────────────────────────────────────────
# Cellpose
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_cellpose(file_bytes: bytes, model_type: str,
                    diameter, flow_threshold: float,
                    cellprob_threshold: float, channels: list):
    """Run Cellpose segmentation. Returns (masks, flows, styles)."""
    from cellpose import models
    gray, rgb = _bytes_to_arrays(file_bytes)
    img = rgb if channels[0] != 0 or channels[1] != 0 else gray
    model = models.Cellpose(model_type=model_type, gpu=False)
    masks, flows, styles, _ = model.eval(
        img,
        diameter=diameter if diameter and diameter > 0 else None,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks, flows, styles, rgb, gray


# ─────────────────────────────────────────────
# Batch processing helper
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_batch_single(file_bytes: bytes, filename: str,
                         # segmentation params
                         seg_method: str, sigma: float, min_size: int,
                         use_watershed: bool,
                         # scale params
                         px_size: float, unit: str,
                         min_area_px: int, max_area_px: int,
                         # focus params
                         focus_metric: str, tile_size: int,
                         blur_percentile: int,
                         # sampling params
                         guard_px: int, exclusion_edges_str: str):
    """
    Full pipeline for a single image: segment → measure → focus → guard frame.
    Returns a flat dict of summary metrics for the batch results table.
    """
    from utils.sampling import segment_objects, apply_guard_frame, measure_all_objects
    from utils.focus import compute_focus_map, global_focus_metrics, flag_blurry_regions
    import numpy as np

    gray, rgb = _bytes_to_arrays(file_bytes)
    h, w = gray.shape

    # Segmentation
    _, labels_raw, n_raw = segment_objects(
        gray, seg_method, sigma, min_size, use_watershed
    )

    # Guard frame
    exclusion_edges = exclusion_edges_str.split(",") if exclusion_edges_str else []
    labels_counted, n_counted, _ = apply_guard_frame(
        labels_raw, h, w, guard_px, exclusion_edges
    )

    # Measurements
    measurements = measure_all_objects(labels_counted, px_size)

    # Focus
    _, tile_scores = compute_focus_map(gray, focus_metric, tile_size)
    blurry_mask = flag_blurry_regions(tile_scores, blur_percentile)
    pct_blurry = 100.0 * blurry_mask.sum() / blurry_mask.size
    g_metrics = global_focus_metrics(gray)

    # Aggregate
    areas = [m["area_um2"] for m in measurements]
    diams = [m["equiv_diameter_um"] for m in measurements]
    circs = [m["circularity"] for m in measurements]

    field_area_um2 = (h * px_size) * (w * px_size)

    return {
        "filename": filename,
        "n_raw": n_raw,
        "n_counted": n_counted,
        f"mean_area_{unit}2": round(float(np.mean(areas)), 3) if areas else 0,
        f"median_area_{unit}2": round(float(np.median(areas)), 3) if areas else 0,
        f"mean_diameter_{unit}": round(float(np.mean(diams)), 3) if diams else 0,
        "mean_circularity": round(float(np.mean(circs)), 3) if circs else 0,
        "density_per_um2": round(n_counted / field_area_um2, 6) if field_area_um2 > 0 else 0,
        "pct_blurry_tiles": round(pct_blurry, 1),
        "laplacian_var": round(g_metrics["laplacian_var"], 1),
        "image_h_px": h,
        "image_w_px": w,
    }
