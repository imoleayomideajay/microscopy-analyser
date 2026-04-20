import numpy as np
from skimage import filters, morphology, measure
from scipy.ndimage import uniform_filter


def get_threshold_methods():
    return ["Otsu", "Li", "Triangle", "Yen", "Sauvola", "Niblack", "Manual"]


def correct_background(gray: np.ndarray, radius: int) -> np.ndarray:
    """Rolling ball background subtraction via morphological opening."""
    selem = morphology.disk(radius)
    background = morphology.opening(gray, selem)
    corrected = gray.astype(np.int32) - background.astype(np.int32)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def apply_threshold(gray: np.ndarray, method: str, sigma: float = 1.0,
                    manual_val: int = None, invert: bool = False):
    """Return (binary_mask, threshold_value_or_method_name)."""
    from skimage.filters import gaussian
    blurred = gaussian(gray, sigma=sigma, preserve_range=True).astype(np.uint8) if sigma > 0 else gray

    method_map = {
        "Otsu": filters.threshold_otsu,
        "Li": filters.threshold_li,
        "Triangle": filters.threshold_triangle,
        "Yen": filters.threshold_yen,
    }

    if method == "Manual":
        thresh = manual_val or 128
        binary = blurred > thresh
        return (~binary if invert else binary), thresh

    if method in ("Sauvola", "Niblack"):
        ws = min(blurred.shape[0], blurred.shape[1], 51)
        if ws % 2 == 0:
            ws -= 1
        if method == "Sauvola":
            thresh_img = filters.threshold_sauvola(blurred, window_size=ws)
        else:
            thresh_img = filters.threshold_niblack(blurred, window_size=ws)
        binary = blurred > thresh_img
        return (~binary if invert else binary), method

    fn = method_map.get(method, filters.threshold_otsu)
    thresh = fn(blurred)
    binary = blurred > thresh
    return (~binary if invert else binary), round(float(thresh), 1)


def segment_and_label(binary: np.ndarray, min_size: int = 50, fill_holes: bool = True):
    cleaned = morphology.remove_small_objects(binary, min_size=max(min_size, 1))
    if fill_holes:
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=max(min_size, 1))
    labeled = measure.label(cleaned)
    return labeled, int(labeled.max())


def overlay_segmentation(rgb: np.ndarray, labeled: np.ndarray, n: int, alpha: float = 0.45):
    """Colour-code each labelled region on the RGB image."""
    import cv2
    overlay = rgb.copy()
    if n == 0:
        return overlay

    np.random.seed(42)
    colors = np.random.randint(80, 230, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    color_img = colors[labeled]
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_img, alpha, 0)

    # Outlines
    for lbl in range(1, n + 1):
        mask = (labeled == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in colors[lbl]), 1)

    return overlay


def compute_segmentation_stats(labeled: np.ndarray, n: int) -> dict:
    total_px = labeled.size
    object_px = (labeled > 0).sum()
    if n == 0:
        return {"coverage_pct": 0.0, "mean_area": 0.0}
    props = measure.regionprops(labeled)
    areas = [p.area for p in props]
    return {
        "coverage_pct": 100.0 * object_px / total_px,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
    }
