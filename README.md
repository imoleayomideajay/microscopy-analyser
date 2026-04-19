# 🔬 MicroAnalyser

A Streamlit application that addresses the **five core problems** in microscopy image analysis for size, shape, and quantification.

## Problems solved

| # | Problem | Module |
|---|---------|--------|
| 1 | **Scale calibration errors** — pixel → physical unit conversion | `pages/scale_calibration.py` |
| 2 | **Poor image segmentation** — adaptive thresholding & background correction | `pages/segmentation.py` |
| 3 | **Touching / overlapping objects** — watershed-based separation | `pages/object_separation.py` |
| 4 | **Focus & depth-of-field artefacts** — local sharpness mapping | `pages/focus_quality.py` |
| 5 | **Sampling & edge-object bias** — guard frame + Abercrombie correction | `pages/sampling_bias.py` |

---

## Quick start (local)

```bash
# 1. Clone
git clone https://github.com/your-username/microscopy-analyser.git
cd microscopy-analyser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done.

No extra configuration needed. `requirements.txt` is picked up automatically.

---

## Project structure

```
microscopy-analyser/
├── app.py                      # Entry point, sidebar routing, global CSS
├── requirements.txt
├── .streamlit/
│   └── config.toml             # Dark theme, port settings
├── pages/
│   ├── __init__.py
│   ├── home.py                 # Landing page
│   ├── scale_calibration.py    # Problem 1
│   ├── segmentation.py         # Problem 2
│   ├── object_separation.py    # Problem 3
│   ├── focus_quality.py        # Problem 4
│   └── sampling_bias.py        # Problem 5
└── utils/
    ├── __init__.py
    ├── image_io.py             # Load images, export helpers
    ├── scale.py                # Scale bar detection, calibration
    ├── segmentation.py         # Thresholding, background correction
    ├── separation.py           # Watershed, label comparison
    ├── focus.py                # Laplacian, Tenengrad, focus maps
    └── sampling.py             # Guard frame, Abercrombie correction
```

---

## Module details

### 1 · Scale calibration (`📐`)
- Three calibration methods: manual µm/pixel, automatic scale bar detection, known reference object
- Renders a calibrated scale bar overlay on the output image
- Exports per-object measurements (area, equivalent diameter, perimeter, circularity) in physical units

### 2 · Segmentation (`✂️`)
- Pre-processing: rolling ball background correction, Gaussian denoising
- Thresholding: Otsu, Li, Triangle, Yen (global); Sauvola, Niblack (local/adaptive)
- Post-processing: small-object removal, hole filling
- Side-by-side view: original / binary mask / labelled overlay

### 3 · Object separation (`🔗`)
- Distance-transform watershed with tunable seed detection (min distance, footprint size, compactness)
- Before/after comparison showing naive vs watershed count
- Seed marker visualisation

### 4 · Focus quality (`🎯`)
- Four sharpness estimators: Laplacian variance, normalised variance, Tenengrad, Brenner gradient
- Tile-based local focus map with configurable tile size
- Blurry tile flagging at user-defined percentile threshold
- Score distribution histogram with threshold line

### 5 · Sampling & edge bias (`📊`)
- Guard frame method: user-selects which edges are exclusion lines
- Abercrombie correction for thick histological sections
- Per-object export with calibrated size and shape descriptors
- Density (objects/µm²) and sampling report

---

## Supported image formats

TIFF (including multi-page — first page used), PNG, JPEG. For best results use 8-bit or 16-bit greyscale TIFFs.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `scikit-image` | Thresholding, watershed, region props |
| `opencv-python-headless` | Image I/O, drawing, background correction |
| `scipy` | Distance transform, label operations |
| `numpy` | Array operations |
| `Pillow` | Image loading |
| `matplotlib` | Focus histograms, colormaps |
| `pandas` | Measurement tables, CSV export |

---

## Contributing

Pull requests welcome. Please open an issue first for major changes.

## Licence

MIT
