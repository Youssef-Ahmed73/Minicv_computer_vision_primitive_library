"""
minicv.features
===============
Feature extraction for images.

Feature descriptors convert an image (or region) into a compact, comparable
numeric vector.  They are a prerequisite for classification, retrieval, and
matching tasks in computer vision.

Contents
--------
Global Descriptors (section 6.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* color_histogram   – multi-channel histogram as a flat feature vector
* pixel_statistics  – mean / std / min / max / skewness per channel

Gradient Descriptors (section 6.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* gradient_histogram – histogram of gradient magnitudes per orientation bin
* lbp                – Local Binary Pattern texture descriptor

All functions return 1-D float64 NumPy arrays suitable for use as feature vectors.
"""

import numpy as np
from .utils import _validate_image, to_float64
from .filters import sobel_gradients


# ===========================================================================
# 6.1  Global Descriptors
# ===========================================================================

def color_histogram(
    img: np.ndarray,
    bins: int = 32,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a concatenated per-channel intensity histogram as a feature vector.

    For a grayscale image  → 1 histogram of `bins` values.
    For an RGB image       → 3 histograms concatenated → vector of 3*bins.

    Parameters
    ----------
    img       : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3),
                             values in [0, 1].
    bins      : int        – number of histogram bins per channel (default 32).
    normalize : bool       – if True, each per-channel histogram is L1-normalised
                             to sum to 1 (so vectors of different image sizes
                             can still be compared).

    Returns
    -------
    np.ndarray float64, shape (bins,) for grayscale or (3*bins,) for RGB.

    Notes
    -----
    Concatenating per-channel histograms is a classic global image descriptor
    used in content-based image retrieval (CBIR).  It is rotation-invariant
    but not scale-invariant.
    """
    _validate_image(img)
    img=to_float64(img)
    def _chan_hist(channel: np.ndarray) -> np.ndarray:
        h, _ = np.histogram(channel.ravel(), bins=bins, range=(0.0, 1.0))
        h = h.astype(np.float64)
        if normalize:
            s = h.sum()
            if s > 0:
                h /= s
        return h

    if img.ndim == 2:
        return _chan_hist(img)

    if img.ndim == 3:
        parts = [_chan_hist(img[:, :, c]) for c in range(img.shape[2])]
        return np.concatenate(parts)

    raise ValueError(f"Image must be 2-D or 3-D, got {img.ndim}-D.")


def pixel_statistics(img: np.ndarray) -> np.ndarray:
    """
    Compute first-order pixel statistics as a compact feature vector.

    For each channel the following statistics are computed:
    mean, standard deviation, minimum, maximum, skewness.

    Skewness formula  (Pearson's moment coefficient)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    μ₃ / σ³   where μ₃ = E[(x − μ)³]

    Parameters
    ----------
    img : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).

    Returns
    -------
    np.ndarray float64 – length 5 for grayscale, length 15 for RGB.
        Each block of 5 values = [mean, std, min, max, skewness].

    Notes
    -----
    These statistics capture the overall brightness, contrast, range, and
    asymmetry of the intensity distribution.  They are very fast to compute
    and useful as a baseline or as a supplement to richer descriptors.
    """
    _validate_image(img)
    
    def _stats(channel: np.ndarray) -> np.ndarray:
        flat = channel.ravel().astype(np.float64)
        mu = flat.mean()
        sigma = flat.std()
        mn = flat.min()
        mx = flat.max()
        if sigma > 0:
            skew = float(np.mean(((flat - mu) / sigma) ** 3))
        else:
            skew = 0.0
        return np.array([mu, sigma, mn, mx, skew], dtype=np.float64)

    if img.ndim == 2:
        return _stats(img)

    if img.ndim == 3:
        parts = [_stats(img[:, :, c]) for c in range(img.shape[2])]
        return np.concatenate(parts)

    raise ValueError(f"Image must be 2-D or 3-D, got {img.ndim}-D.")


# ===========================================================================
# 6.2  Gradient Descriptors
# ===========================================================================

def gradient_histogram(
    img: np.ndarray,
    bins: int = 9,
    *,
    signed: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a magnitude-weighted histogram of gradient orientations.

    This is the core idea behind HOG (Histogram of Oriented Gradients) applied
    globally to the whole image rather than per-cell.

    Algorithm
    ---------
    1. Compute Sobel gradients (Gx, Gy) → magnitude M, angle θ.
    2. Quantise θ into *bins* equal-width orientation bins.
    3. Accumulate gradient magnitude M into the corresponding bin
       (magnitude weighting means stronger edges contribute more).
    4. Optionally L2-normalise the resulting vector.

    Parameters
    ----------
    img      : np.ndarray – 2-D float64 grayscale image.
    bins     : int        – number of orientation bins (default 9, as in HOG).
    signed   : bool       – if False, angles in [0°, 180°) (unsigned);
                            if True, angles in [0°, 360°) (signed).
    normalize: bool       – L2-normalise the output vector.

    Returns
    -------
    np.ndarray float64, shape (bins,) – orientation histogram.

    Raises
    ------
    ValueError – img is not 2-D.
    """
    _validate_image(img, ndim=2)
    grads = sobel_gradients(img.astype(np.float64))
    mag = grads["magnitude"]
    ang = grads["angle"]  # radians, range (-π, π)

    # Convert to degrees; map to [0°, 360°) or [0°, 180°)
    ang_deg = np.rad2deg(ang) % 360.0
    if not signed:
        ang_deg = ang_deg % 180.0

    upper = 360.0 if signed else 180.0
    hist, _ = np.histogram(ang_deg, bins=bins, range=(0.0, upper), weights=mag)
    hist = hist.astype(np.float64)

    if normalize:
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

    return hist


def lbp(
    img: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
    *,
    bins: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the Local Binary Pattern (LBP) texture descriptor.

    LBP encodes the local texture around each pixel by comparing each of
    *n_points* equally-spaced neighbours on a circle of *radius* to the
    centre pixel.  The comparison results form a binary code.

    Algorithm
    ---------
    For each pixel p at (y, x):
        1. Sample *n_points* neighbours at angles  θ_k = 2π·k / n_points
           on a circle of radius *radius* (bilinear interpolation).
        2. Compare each neighbour to the centre:  b_k = 1 if nbr_k ≥ p.
        3. Form the code:  LBP(y,x) = Σ_k  b_k · 2^k.
    Then compute the histogram of all LBP codes over the image.

    Parameters
    ----------
    img       : np.ndarray – 2-D float64 grayscale.
    radius    : int        – radius of the sampling circle (> 0).
    n_points  : int        – number of sampling points on the circle.
    bins      : int        – number of histogram bins (≤ 2^n_points).
    normalize : bool       – L1-normalise the histogram.

    Returns
    -------
    np.ndarray float64, shape (bins,) – LBP histogram feature vector.

    Raises
    ------
    ValueError – img is not 2-D or radius ≤ 0.

    Notes
    -----
    LBP is invariant to monotonic illumination changes (it only uses relative
    comparisons).  It is widely used for face recognition and texture analysis.
    The circular sampling with bilinear interpolation is the standard uniform
    LBP variant (Ojala et al., 2002).
    """
    _validate_image(img, ndim=2)
    if radius <= 0:
        raise ValueError(f"'radius' must be > 0, got {radius}.")

    src = img.astype(np.float64)
    H, W = src.shape
    codes = np.zeros((H, W), dtype=np.float64)

    # Precompute neighbour offsets (vectorised over positions, loop over points)
    # Loop justification: we iterate over n_points (≤ 24 typically) sample angles,
    # each iteration is a fully vectorised operation over all HxW pixels.
    angles = np.array([2 * np.pi * k / n_points for k in range(n_points)])
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)

    for k, theta in enumerate(angles):
        dy = -radius * np.sin(theta)   # row offset (up is negative)
        dx = radius * np.cos(theta)    # col offset

        nbr_y = rr + dy
        nbr_x = cc + dx

        # Bilinear sampling of neighbour; clamp to border
        y0 = np.clip(np.floor(nbr_y).astype(int), 0, H - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        x0 = np.clip(np.floor(nbr_x).astype(int), 0, W - 1)
        x1 = np.clip(x0 + 1, 0, W - 1)
        fy = nbr_y - np.floor(nbr_y)
        fx = nbr_x - np.floor(nbr_x)

        nbr_val = (
            (1 - fy) * (1 - fx) * src[y0, x0]
            + (1 - fy) * fx * src[y0, x1]
            + fy * (1 - fx) * src[y1, x0]
            + fy * fx * src[y1, x1]
        )

        # Threshold comparison; accumulate binary code
        codes += (nbr_val >= src).astype(np.float64) * (2 ** k)

    # Histogram of LBP codes
    max_code = 2 ** n_points
    hist, _ = np.histogram(codes.ravel(), bins=bins, range=(0, max_code))
    hist = hist.astype(np.float64)

    if normalize:
        s = hist.sum()
        if s > 0:
            hist /= s

    return hist
