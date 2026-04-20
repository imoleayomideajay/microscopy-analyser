# 🔬 MicroAnalyser

A Streamlit application that solves the **five core problems** in microscopy image analysis for size, shape, and quantification — extended with deep-learning segmentation (Cellpose), batch processing, and full computation caching.

---

## Modules

| # | Module | Key techniques |
|---|--------|---------------|
| 1 | 📐 **Scale Calibration** | Scale bar detection, manual µm/px, known-object method |
| 2 | ✂️ **Segmentation** | Otsu, Li, Triangle, Yen, Sauvola, Niblack; rolling ball BG correction |
| 3 | 🔗 **Object Separation** | Distance-transform watershed, tunable seed detection |
| 4 | 🎯 **Focus Quality** | Laplacian variance, Tenengrad, Brenner, normalised variance; tile heatmap |
| 5 | 📊 **Sampling & Edge Bias** | Guard frame method, Abercrombie correction, density report |
| ✨ | 🧬 **Cellpose (DL)** | cyto3 / nuclei / tissuenet / livecell neural network models |
| ✨ | 🗂️ **Batch Processing** | Full pipeline across N images → consolidated CSV + ZIP export |

All modules use `@st.cache_data` — slider adjustments reuse cached intermediates for near-instant response.

---

## Quick start

### Local

```bash
git clone https://github.com/your-username/microscopy-analyser.git
cd microscopy-analyser
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

### Streamlit Community Cloud (free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, main file `app.py`
4. Click **Deploy** — `requirements.txt` is picked up automatically

### Docker (lab server / self-hosted)

```bash
docker build -t microanalyser .
docker run -p 8501:8501 microanalyser
# → http://localhost:8501
```

---

## Project structure

```
microscopy-analyser/
├── app.py                       # Entry point, sidebar routing, global CSS
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── config.toml              # Dark theme, port
├── pages/
│   ├── home.py                  # Landing page
│   ├── scale_calibration.py     # Problem 1
│   ├── segmentation.py          # Problem 2
│   ├── object_separation.py     # Problem 3
│   ├── focus_quality.py         # Problem 4
│   ├── sampling_bias.py         # Problem 5
│   ├── cellpose_seg.py          # Deep-learning segmentation
│   └── batch_processing.py      # Multi-image batch pipeline
└── utils/
    ├── image_io.py              # Load images, export helpers
    ├── scale.py                 # Scale bar detection, calibration, measurement
    ├── segmentation.py          # Thresholding, background correction
    ├── separation.py            # Watershed, label comparison
    ├── focus.py                 # Sharpness estimators, focus maps
    ├── sampling.py              # Guard frame, Abercrombie, density
    └── cache.py                 # @st.cache_data wrappers for all heavy ops
```

---

## Module details

### 🧬 Cellpose (deep learning)
- Models: `cyto3` (general), `nuclei`, `tissuenet`, `livecell`
- Channel configuration for single- and multi-channel fluorescence
- Tunable: diameter, flow threshold, cell probability threshold
- Outputs: labelled overlay, colour mask, per-cell measurements (area, diameter, circularity, eccentricity, solidity, mean intensity), distribution plots
- Gracefully degrades with an install prompt when Cellpose is not present
- Weights (~200 MB) downloaded once on first use, then cached by Cellpose internally

### 🗂️ Batch processing
- Upload any number of images at once
- Configurable pipeline: segmentation method → scale → focus metric → guard frame
- Per-image summary row: object count (raw + guard-frame corrected), mean/median area, mean diameter, mean circularity, density (objects/µm²), blurry tile %, Laplacian variance, image dimensions, processing time
- Blurry images flagged in table (red row highlight) and optionally skipped
- Summary plots: objects per image, mean area per image, blur % bar chart, density vs circularity scatter coloured by blur
- Export: CSV or ZIP (CSV + `pipeline_params.json` for reproducibility)

### ⚡ Caching architecture
All computationally expensive functions in `utils/` are wrapped in `utils/cache.py` using `@st.cache_data`. The cache key includes all parameters that affect the output, so:
- Moving a slider only recomputes what changed
- Switching tabs doesn't re-run segmentation
- The same image uploaded twice is processed only once per parameter set

---

## Supported formats

TIFF (including 16-bit and multi-page — first frame used), PNG, JPEG.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `scikit-image` | Thresholding, watershed, regionprops |
| `opencv-python-headless` | Drawing, background correction, contours |
| `scipy` | Distance transform, labelling |
| `numpy` | Array operations |
| `Pillow` | Image loading |
| `matplotlib` | Histograms, colormaps, batch plots |
| `pandas` | Measurement tables, CSV export |
| `cellpose` | Deep-learning cell segmentation |

---

## Licence

MIT
