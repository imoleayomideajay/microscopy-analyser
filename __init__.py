import numpy as np
import cv2
from skimage import measure, morphology, filters


def detect_scale_bar(gray: np.ndarray, known_length: float):
    """
    Very simple scale bar detector: looks for a thick horizontal white/black
    bar in the bottom 15% of the image. Returns (µm_per_pixel, rect) or (None, None).
    """
    h, w = gray.shape
    roi = gray[int(0.85 * h):, :]

    # Threshold to binary
    thresh = filters.threshold_otsu(roi)
    binary = (roi > thresh).astype(np.uint8)

    # Find horizontal runs
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)

    best = None
    for p in props:
        r0, c0, r1, c1 = p.bbox
        width = c1 - c0
        height = r1 - r0
        aspect = width / max(height, 1)
        if aspect > 8 and width > 30:  # long and thin
            if best is None or width > best.major_axis_length:
                best = p

    if best is None:
        return None, None

    bar_px = best.bbox[3] - best.bbox[1]
    px_per_unit = bar_px / known_length
    return 1.0 / px_per_unit, best.bbox


def apply_calibration(value_px: float, px_size: float) -> float:
    return value_px * px_size


def measure_objects_calibrated(gray, rgb, px_size, unit, min_area_px, max_area_px):
    """Segment and return calibrated measurements."""
    from skimage.filters import threshold_otsu, gaussian
    from skimage import morphology, measure

    blurred = gaussian(gray, sigma=1.5, preserve_range=True).astype(np.uint8)
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=min_area_px)
    binary = morphology.remove_small_holes(binary, area_threshold=min_area_px)

    labeled = measure.label(binary)
    props = measure.regionprops(labeled, intensity_image=gray)

    measurements = []
    overlay = rgb.copy()
    colors = _label_colors(labeled.max())

    for p in props:
        area_px = p.area
        if not (min_area_px <= area_px <= max_area_px):
            continue

        area_cal = area_px * (px_size ** 2)
        diam_cal = p.equivalent_diameter_area * px_size
        perim_cal = p.perimeter * px_size
        circ = (4 * np.pi * area_px) / (p.perimeter ** 2 + 1e-9)

        measurements.append({
            "label": p.label,
            f"area_{unit}2": round(area_cal, 4),
            f"equiv_diameter_{unit}": round(diam_cal, 4),
            f"perimeter_{unit}": round(perim_cal, 4),
            "circularity": round(circ, 4),
            "eccentricity": round(p.eccentricity, 4),
            "solidity": round(p.solidity, 4),
            "mean_intensity": round(float(p.mean_intensity), 2),
        })

        # Draw outline on overlay
        color = colors[p.label % len(colors)]
        contour_mask = np.zeros_like(gray, dtype=np.uint8)
        contour_mask[labeled == p.label] = 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay, measurements


def draw_scale_overlay(rgb, px_size, unit, bar_um=10.0):
    """Draw a calibrated scale bar on the image."""
    h, w = rgb.shape[:2]
    bar_px = int(bar_um / px_size)
    bar_px = max(10, min(bar_px, w // 3))

    margin = 15
    y = h - margin - 8
    x_end = w - margin
    x_start = x_end - bar_px

    # White bar with black outline
    cv2.rectangle(rgb, (x_start - 1, y - 1), (x_end + 1, y + 9), (0, 0, 0), -1)
    cv2.rectangle(rgb, (x_start, y), (x_end, y + 8), (255, 255, 255), -1)

    label = f"{bar_um:.0f} {unit}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.35
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    tx = x_start + (bar_px - tw) // 2
    cv2.putText(rgb, label, (tx, y - 3), font, fs, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(rgb, label, (tx, y - 3), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return rgb


def _label_colors(n):
    np.random.seed(42)
    return [tuple(int(c) for c in np.random.randint(80, 240, 3)) for _ in range(max(n + 1, 2))]
