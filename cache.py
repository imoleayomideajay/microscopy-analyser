import numpy as np
import cv2
from skimage.filters import laplace


def _laplacian_variance(patch):
    lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _normalised_variance(patch):
    m = patch.mean()
    if m < 1e-6:
        return 0.0
    return float(patch.var() / m)


def _tenengrad(patch):
    gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = gx ** 2 + gy ** 2
    return float(g.mean())


def _brenner(patch):
    diff = patch.astype(np.float32)
    shifted = np.roll(diff, -2, axis=1)
    return float(((shifted - diff) ** 2).mean())


_METRICS = {
    "Laplacian variance": _laplacian_variance,
    "Normalised variance": _normalised_variance,
    "Tenengrad (Sobel gradient energy)": _tenengrad,
    "Brenner gradient": _brenner,
}


def compute_focus_map(gray: np.ndarray, metric_name: str, tile_size: int):
    fn = _METRICS.get(metric_name, _laplacian_variance)
    h, w = gray.shape
    rows = max(1, h // tile_size)
    cols = max(1, w // tile_size)

    scores = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * tile_size:(r + 1) * tile_size,
                         c * tile_size:(c + 1) * tile_size]
            scores[r, c] = fn(patch)

    # Normalise to 0-1
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-9)

    # Upscale to original image size
    focus_map = cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST)
    return focus_map, scores


def flag_blurry_regions(tile_scores: np.ndarray, percentile: int):
    threshold = np.percentile(tile_scores, percentile)
    return tile_scores < threshold


def global_focus_metrics(gray: np.ndarray) -> dict:
    lap_var = _laplacian_variance(gray)
    tg = _tenengrad(gray)
    return {"laplacian_var": lap_var, "tenengrad": tg}


def annotate_focus_map(rgb, focus_map, blurry_mask, tile_size, cmap_name, show_grid, show_flagged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    heat = (cmap(focus_map)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 0.45, heat, 0.55, 0)

    h, w = rgb.shape[:2]
    rows, cols = blurry_mask.shape

    if show_grid:
        for r in range(rows + 1):
            y = min(r * tile_size, h - 1)
            cv2.line(overlay, (0, y), (w, y), (30, 30, 50), 1)
        for c in range(cols + 1):
            x = min(c * tile_size, w - 1)
            cv2.line(overlay, (x, 0), (x, h), (30, 30, 50), 1)

    if show_flagged:
        for r in range(rows):
            for c in range(cols):
                if blurry_mask[r, c]:
                    y0, x0 = r * tile_size, c * tile_size
                    y1 = min(y0 + tile_size, h)
                    x1 = min(x0 + tile_size, w)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 60, 60), 1)

    return overlay
