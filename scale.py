import numpy as np
import cv2
from skimage import filters, morphology, measure, segmentation, feature
from scipy import ndimage as ndi


def segment_objects(gray, method="Otsu", sigma=1.5, min_size=80, use_watershed=True):
    from skimage.filters import gaussian
    blurred = gaussian(gray, sigma=sigma, preserve_range=True).astype(np.uint8)
    method_map = {
        "Otsu": filters.threshold_otsu,
        "Li": filters.threshold_li,
        "Triangle": filters.threshold_triangle,
        "Yen": filters.threshold_yen,
    }
    fn = method_map.get(method, filters.threshold_otsu)
    thresh = fn(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=max(min_size, 1))
    binary = morphology.remove_small_holes(binary, area_threshold=max(min_size, 1))

    if use_watershed:
        distance = ndi.distance_transform_edt(binary)
        local_max = feature.peak_local_max(distance, min_distance=10, labels=binary)
        markers_mask = np.zeros(distance.shape, dtype=bool)
        if len(local_max) > 0:
            markers_mask[tuple(local_max.T)] = True
        markers, _ = ndi.label(markers_mask)
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labels = measure.label(binary)

    return binary, labels, int(labels.max())


def apply_guard_frame(labels, h, w, guard_px, exclusion_edges):
    """
    Guard frame counting rule.
    Inclusion lines: bottom, right edges of guard frame.
    Exclusion lines: top, left (or user-selected).
    """
    edge_map = {
        "Top": (0, guard_px, 0, w),          # rows 0..guard_px
        "Bottom": (h - guard_px, h, 0, w),
        "Left": (0, h, 0, guard_px),
        "Right": (0, h, w - guard_px, w),
    }

    props = measure.regionprops(labels)
    excluded = set()

    for p in props:
        r0, c0, r1, c1 = p.bbox
        for edge in exclusion_edges:
            er0, er1, ec0, ec1 = edge_map[edge]
            # Does this object's bbox overlap with the exclusion strip?
            if r0 < er1 and r1 > er0 and c0 < ec1 and c1 > ec0:
                excluded.add(p.label)
                break

    new_labels = labels.copy()
    for lbl in excluded:
        new_labels[labels == lbl] = 0

    # Re-label sequentially
    new_labels = measure.label(new_labels > 0)
    return new_labels, int(new_labels.max()), excluded


def abercrombie_correction(n_counted, section_thickness, object_height, field_h_um, field_w_um):
    """N_true = N_counted * T / (T + h)"""
    return n_counted * section_thickness / (section_thickness + object_height)


def measure_all_objects(labels, px_size):
    props = measure.regionprops(labels)
    results = []
    for p in props:
        area_um = p.area * (px_size ** 2)
        diam_um = p.equivalent_diameter_area * px_size
        perim_um = p.perimeter * px_size
        circ = (4 * np.pi * p.area) / (p.perimeter ** 2 + 1e-9)
        results.append({
            "label": p.label,
            "area_um2": round(area_um, 4),
            "equiv_diameter_um": round(diam_um, 4),
            "perimeter_um": round(perim_um, 4),
            "circularity": round(circ, 4),
            "eccentricity": round(p.eccentricity, 4),
            "solidity": round(p.solidity, 4),
        })
    return results


def visualise_guard_frame(rgb, labels_counted, labels_raw, guard_px, exclusion_edges):
    h, w = rgb.shape[:2]
    overlay = rgb.copy()

    # Colour counted objects green, excluded red
    counted_set = set(np.unique(labels_counted)) - {0}
    raw_props = measure.regionprops(labels_raw)

    np.random.seed(42)
    for p in raw_props:
        mask = (labels_raw == p.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any counted label overlaps
        overlap = labels_counted[labels_raw == p.label]
        is_counted = any(v > 0 for v in overlap)
        color = (60, 220, 100) if is_counted else (220, 60, 60)
        cv2.drawContours(overlay, contours, -1, color, 2)

    # Draw guard frame
    if guard_px > 0:
        edge_map = {
            "Top": [(0, 0), (w, 0), (w, guard_px), (0, guard_px)],
            "Bottom": [(0, h - guard_px), (w, h - guard_px), (w, h), (0, h)],
            "Left": [(0, 0), (guard_px, 0), (guard_px, h), (0, h)],
            "Right": [(w - guard_px, 0), (w, 0), (w, h), (w - guard_px, h)],
        }
        # Draw inner frame boundary in blue
        pts = [
            (guard_px, guard_px),
            (w - guard_px, guard_px),
            (w - guard_px, h - guard_px),
            (guard_px, h - guard_px),
        ]
        cv2.polylines(overlay, [np.array(pts)], True, (80, 140, 255), 2)

    return overlay


def sampling_report(n_raw, n_counted, n_corrected, measurements, field_h_um, field_w_um, px_size):
    field_area = field_h_um * field_w_um
    density = n_counted / field_area if field_area > 0 else 0
    return {
        "n_raw": n_raw,
        "n_counted": n_counted,
        "n_corrected": n_corrected,
        "field_area_um2": field_area,
        "density": density,
    }
