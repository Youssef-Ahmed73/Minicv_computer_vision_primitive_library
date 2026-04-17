"""
minicv.io
=========
Image I/O and colour-space conversion.

Why use Matplotlib for I/O?
---------------------------
The project constraints forbid PIL/Pillow and OpenCV.  Matplotlib's
``imread`` / ``imsave`` support PNG natively on every platform via its
bundled ``Agg`` renderer, and JPG support is available when Pillow is
present.  The functions below wrap those calls with explicit validation
and dtype normalisation so the rest of minicv always receives consistent
float64 arrays.

Contents
--------
* read_image   – load from disk → float64 ndarray
* export_image – save ndarray   → PNG / JPG on disk
* rgb_to_gray  – colour → luminance-weighted grayscale
* gray_to_rgb  – grayscale → 3-channel (replicated) RGB
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from .utils import _validate_image, to_float64, to_uint8


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_image(path: str | Path) -> np.ndarray:
    """
    Load an image file from disk into a NumPy array.

    Supported formats depend on the active Matplotlib backend:
    PNG is always supported; JPG requires Pillow.

    Parameters
    ----------
    path : str | Path – file path to the image.

    Returns
    -------
    np.ndarray float64
        * Grayscale PNG → shape (H, W),   values in [0, 1].
        * RGB  image    → shape (H, W, 3), values in [0, 1].
        * RGBA image    → alpha channel dropped, shape (H, W, 3).

    Raises
    ------
    FileNotFoundError – path does not exist.
    ValueError        – file cannot be decoded as an image.

    Notes
    -----
    Matplotlib's ``imread`` returns float32 for PNG (values 0–1) and
    uint8 for JPG (values 0–255).  We normalise both to float64 [0, 1]
    so callers never need to handle the difference.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: '{path}'.")

    try:
        raw = mpimg.imread(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read '{path}' as an image: {exc}") from exc

    # Normalise to float64 [0, 1]
    #img = to_float64(raw)
    img=raw
    # Drop alpha channel if present (H, W, 4) → (H, W, 3)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_image(
    img: np.ndarray,
    path: str | Path,
    *,
    quality: int = 95,
) -> None:
    """
    Save a NumPy array to an image file on disk.

    Parameters
    ----------
    img     : np.ndarray – 2-D (H, W) grayscale or 3-D (H, W, 3) RGB, float64
              in [0, 1] or uint8 in [0, 255].
    path    : str | Path – output file path.  Extension determines format
              (.png, .jpg / .jpeg).
    quality : int        – JPEG quality 1–95 (only used for .jpg/.jpeg).

    Returns
    -------
    None  (file is written to disk).

    Raises
    ------
    TypeError  – img is not ndarray.
    ValueError – img has wrong ndim, or unsupported extension.

    Notes
    -----
    Matplotlib's ``imsave`` accepts float arrays in [0, 1] directly.
    For grayscale we pass ``cmap='gray'`` to prevent false-colour output.
    """
    _validate_image(img)

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2-D (grayscale) or 3-D (RGB), got {img.ndim}-D."
        )

    path = Path(path)
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .png, .jpg, or .jpeg."
        )

    # Convert to float64 [0,1] for imsave
    if img.dtype == np.uint8:
        out = img.astype(np.float64) / 255.0
    else:
        out = np.clip(img.astype(np.float64), 0.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {}
    if ext in {".jpg", ".jpeg"}:
        kwargs["pil_kwargs"] = {"quality": quality}

    if out.ndim == 2:
        mpimg.imsave(str(path), out, cmap="gray", vmin=0, vmax=1, **kwargs)
    else:
        mpimg.imsave(str(path), out, **kwargs)


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using ITU-R BT.601 luminance weights.

    Formula
    -------
    Y = 0.2989 · R + 0.5870 · G + 0.1140 · B

    These weights account for human perception: the eye is most sensitive to
    green, moderately to red, and least to blue.  They are the same weights
    used by OpenCV's ``cvtColor(BGR2GRAY)`` (with channels reordered).

    Parameters
    ----------
    img : np.ndarray – shape (H, W, 3), float64 in [0, 1].

    Returns
    -------
    np.ndarray float64, shape (H, W), values in [0, 1].

    Raises
    ------
    TypeError  – img is not ndarray.
    ValueError – img is not 3-D with 3 channels.
    """
    _validate_image(img, ndim=3)
    img = to_float64(img)
    if img.shape[2] != 3:
        raise ValueError(
            f"Expected 3 colour channels, got {img.shape[2]}."
        )
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    return (img.astype(np.float64) @ weights)


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to a 3-channel RGB image by replicating the channel.

    Parameters
    ----------
    img : np.ndarray – shape (H, W), any numeric dtype.

    Returns
    -------
    np.ndarray float64, shape (H, W, 3), where R == G == B == luminance.

    Raises
    ------
    TypeError  – img is not ndarray.
    ValueError – img is not 2-D.

    Notes
    -----
    ``np.stack`` with three copies is used rather than ``np.repeat`` to keep
    the output contiguous in memory, which matters for downstream JPEG export.
    """
    _validate_image(img, ndim=2)
    f = img.astype(np.float64)
    return np.stack([f, f, f], axis=2)
