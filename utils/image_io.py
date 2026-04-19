import numpy as np
from PIL import Image
import io


def load_image(uploaded_file):
    """Load an uploaded file → (gray uint8 ndarray, RGB uint8 ndarray)."""
    data = uploaded_file.read()
    pil = Image.open(io.BytesIO(data))

    # Handle multi-page TIFFs — take first page
    try:
        pil.seek(0)
    except Exception:
        pass

    # Convert to RGB
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    elif pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")

    rgb = np.array(pil.convert("RGB"), dtype=np.uint8)
    gray = np.array(pil.convert("L"), dtype=np.uint8)
    return gray, rgb


def make_download_bytes(img_rgb: np.ndarray, fmt="PNG") -> bytes:
    """Convert RGB ndarray → bytes for st.download_button."""
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()
