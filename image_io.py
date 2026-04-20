import numpy as np
from skimage import filters, morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import cv2


def segment_binary(gray, method="Otsu", sigma=1.5, min_size=80):
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
    return binary


def apply_watershed(binary, gray, min_distance=10, compactness=0.0, footprint_size=7):
    """Distance-transform watershed with local maxima as seeds."""
    distance = ndi.distance_transform_edt(binary)

    fp = np.ones((footprint_size, footprint_size), dtype=bool)
    local_max_mask = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        footprint=fp,
        labels=binary,
    )

    # Convert coordinate array to mask
    markers_mask = np.zeros(distance.shape, dtype=bool)
    markers_mask[tuple(local_max_mask.T)] = True
    markers, _ = ndi.label(markers_mask)

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=binary,
        compactness=compactness,
    )

    return labels, int(labels.max()), local_max_mask


def overlay_labels(rgb, labeled, alpha=0.45, markers=None):
    n = int(labeled.max())
    if n == 0:
        return rgb

    np.random.seed(42)
    colors = np.random.randint(60, 230, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    color_img = colors[labeled]
    out = cv2.addWeighted(rgb, 1 - alpha, color_img, alpha, 0)

    for lbl in range(1, n + 1):
        mask = (labeled == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in colors[lbl]), 1)

    # Draw seed markers
    if markers is not None and len(markers) > 0:
        for pt in markers:
            cv2.circle(out, (int(pt[1]), int(pt[0])), 2, (255, 255, 255), -1)

    return out


def compare_counts(labels_naive, labels_ws):
    props_n = measure.regionprops(labels_naive)
    props_w = measure.regionprops(labels_ws)

    def _stats(props, name):
        if not props:
            return {"source": name, "count": 0, "mean_area_px2": 0, "median_area_px2": 0}
        areas = [p.area for p in props]
        return {
            "source": name,
            "count": len(props),
            "mean_area_px2": round(float(np.mean(areas)), 1),
            "median_area_px2": round(float(np.median(areas)), 1),
        }

    return [_stats(props_n, "Naive"), _stats(props_w, "Watershed")]
