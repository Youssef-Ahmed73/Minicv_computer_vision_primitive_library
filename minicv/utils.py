"""
minicv.utils
============
Low-level building blocks shared across every other module.

Why a utils module?
-------------------
OpenCV itself has dozens of internal helpers (type checks, border handling,
saturation arithmetic, …).  Centralising them here means every other module
can import what it needs without duplicating code, and the logic only needs
to be unit-tested once.

Contents
--------
* _validate_image          – ensure ndarray with correct ndim/dtype
* _validate_kernel         – ensure kernel is numeric, odd-shaped, non-empty
* to_float64               – safe cast to float64
* to_uint8                 – saturate-cast back to uint8
* normalize                – min-max / z-score / unit-norm normalisation
* clip_pixels              – clamp pixel values to [lo, hi]
* pad                      – constant / reflect / replicate (wrap) padding
* convolve2d               – true 2-D correlation (grayscale)
* spatial_filter           – apply a kernel to grayscale or RGB image
"""

import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_image(img, *, name: str = "img", ndim: int | None = None) -> None:
    """
    Raise TypeError / ValueError when *img* is not a valid image array.

    Parameters
    ----------
    img   : object  – the value to inspect.
    name  : str     – variable name used in error messages.
    ndim  : int | None – if given, assert img.ndim == ndim.

    Raises
    ------
    TypeError  – img is not a numpy ndarray.
    ValueError – img has wrong number of dimensions.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"'{name}' must be a numpy.ndarray, got {type(img).__name__}."
        )
    if ndim is not None and img.ndim != ndim:
        raise ValueError(
            f"'{name}' must be {ndim}-D, got {img.ndim}-D."
        )
    if img.size == 0:
        raise ValueError(f"'{name}' must not be empty.")


def _validate_kernel(k: np.ndarray, *, name: str = "kernel") -> None:
    """
    Ensure *k* is a valid 2-D convolution kernel.

    Rules (matching OpenCV conventions)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ndarray with numeric dtype
    * exactly 2-D
    * non-empty
    * both dimensions must be odd (so there is a well-defined centre pixel)

    Raises
    ------
    TypeError  – k is not ndarray or dtype is not numeric.
    ValueError – k is not 2-D, is empty, or has even-sized dimension.
    """
    if not isinstance(k, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(k).__name__}.")
    if not np.issubdtype(k.dtype, np.number):
        raise TypeError(f"'{name}' must have a numeric dtype, got {k.dtype}.")
    if k.ndim != 2:
        raise ValueError(f"'{name}' must be 2-D, got {k.ndim}-D.")
    if k.size == 0:
        raise ValueError(f"'{name}' must not be empty.")
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError(
            f"'{name}' dimensions must both be odd, got shape {k.shape}."
        )


# ---------------------------------------------------------------------------
# dtype conversions
# ---------------------------------------------------------------------------

def to_float64(img: np.ndarray) -> np.ndarray:
    """
    Cast *img* to float64.

    If *img* is uint8 the values are scaled to [0, 1].
    Any other numeric dtype is cast directly (no rescaling).

    Parameters
    ----------
    img : np.ndarray – input image (any numeric dtype).

    Returns
    -------
    np.ndarray with dtype float64.

    Notes
    -----
    Storing images as float64 in [0, 1] is the canonical internal
    representation used throughout minicv, matching scikit-image convention.
    """
    _validate_image(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Saturate-cast *img* to uint8.

    Values are first clipped to [0, 1] if the array is float (assumed [0,1]
    range), then scaled to [0, 255] and cast.  Integer arrays outside
    [0, 255] are clipped directly.

    Parameters
    ----------
    img : np.ndarray – float ([0,1]) or integer image.

    Returns
    -------
    np.ndarray with dtype uint8.
    """
    _validate_image(img)
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(
    img: np.ndarray,
    mode: str = "minmax",
    *,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """
    Normalize pixel values in *img*.

    Parameters
    ----------
    img  : np.ndarray – input image (grayscale or RGB), any numeric dtype.
    mode : str        – one of ``'minmax'``, ``'zscore'``, ``'unit'``.
    mean : float | None – override mean for zscore (computed from data if None).
    std  : float | None – override std  for zscore (computed from data if None).

    Returns
    -------
    np.ndarray float64 with normalised values.

    Raises
    ------
    ValueError – unknown mode, or std == 0 in zscore mode.

    Modes
    -----
    minmax
        x' = (x − min) / (max − min)   →  range [0, 1]
        Preserves relative pixel differences; handles constant images by
        returning all-zeros.

    zscore
        x' = (x − μ) / σ               →  zero mean, unit variance
        Useful as a preprocessing step before feeding images to ML models.
        If the image is constant σ=0; raises ValueError to avoid NaN output.

    unit
        x' = x / ‖x‖₂                  →  ‖x'‖₂ = 1
        Treats the entire image as a flat vector and normalises its L2 norm.
        Common in feature-vector normalisation for descriptors.
    """
    _validate_image(img)
    valid = {"minmax", "zscore", "unit"}
    if mode not in valid:
        raise ValueError(f"'mode' must be one of {valid}, got '{mode}'.")

    f = img.astype(np.float64)

    if mode == "minmax":
        lo, hi = f.min(), f.max()
        if hi == lo:
            return np.zeros_like(f)
        return (f - lo) / (hi - lo)

    if mode == "zscore":
        mu = mean if mean is not None else f.mean()
        sigma = std if std is not None else f.std()
        if sigma == 0:
            raise ValueError(
                "Cannot apply zscore normalisation: image has zero standard deviation "
                "(all pixels are the same value)."
            )
        return (f - mu) / sigma

    # mode == "unit"
    norm = np.linalg.norm(f)
    if norm == 0:
        raise ValueError(
            "Cannot apply unit normalisation: image L2-norm is zero (all pixels are 0)."
        )
    return f / norm


# ---------------------------------------------------------------------------
# Pixel clipping
# ---------------------------------------------------------------------------

def clip_pixels(
    img: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """
    Clamp pixel values of *img* to the closed interval [lo, hi].

    Parameters
    ----------
    img : np.ndarray – input image (any numeric dtype).
    lo  : float      – lower bound (default 0).
    hi  : float      – upper bound (default 1).

    Returns
    -------
    np.ndarray with same dtype as *img* and all values in [lo, hi].

    Raises
    ------
    ValueError – lo >= hi.

    Notes
    -----
    This is a thin wrapper around np.clip; its value is as a named,
    validated entry-point so callers don't hard-code magic numbers.
    """
    _validate_image(img)
    if lo >= hi:
        raise ValueError(f"'lo' ({lo}) must be strictly less than 'hi' ({hi}).")
    return np.clip(img, lo, hi)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad(
    img: np.ndarray,
    pad_h: int,
    pad_w: int,
    mode: str = "constant",
    *,
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Add a border of *pad_h* rows and *pad_w* columns around *img*.

    Parameters
    ----------
    img            : np.ndarray – 2-D (H, W) or 3-D (H, W, C) image.
    pad_h          : int        – number of rows to add on each of top & bottom.
    pad_w          : int        – number of columns to add on each of left & right.
    mode           : str        – ``'constant'``, ``'reflect'``, or ``'replicate'``.
    constant_value : float      – fill value used when mode == 'constant'.

    Returns
    -------
    np.ndarray with shape (H + 2*pad_h, W + 2*pad_w[, C]).

    Raises
    ------
    TypeError  – img is not ndarray.
    ValueError – mode is unknown, or pad amounts are negative.

    Modes
    -----
    constant
        Fill border with a fixed scalar (default 0).  Zero-padding is the
        default choice for convolutions when you want to preserve spatial size.

    reflect
        Mirror pixels at the boundary without repeating the edge pixel.
        E.g. border [d c b | a b c d | c b a].
        Good for images with smooth content near the borders.

    replicate
        Repeat the outermost pixel column/row indefinitely.
        E.g. border [a a a | a b c d | d d d].
        Avoids dark/bright artefacts at image edges.
    """
    _validate_image(img)
    valid_modes = {"constant", "reflect", "replicate"}
    if mode not in valid_modes:
        raise ValueError(f"'mode' must be one of {valid_modes}, got '{mode}'.")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"pad_h and pad_w must be >= 0, got ({pad_h}, {pad_w}).")

    if mode == "constant":
        if img.ndim == 2:
            return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                          mode="constant", constant_values=constant_value)
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                      mode="constant", constant_values=constant_value)

    if mode == "reflect":
        np_mode = "reflect"
    else:  # replicate → 'edge' in numpy
        np_mode = "edge"

    if img.ndim == 2:
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=np_mode)
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=np_mode)


# ---------------------------------------------------------------------------
# 2-D Convolution (cross-correlation)
# ---------------------------------------------------------------------------

def convolve2d(
    img: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Apply a 2-D kernel to a **grayscale** image using true convolution.

    Implementation note — correlation vs convolution
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    OpenCV ``filter2D`` performs **cross-correlation**, not strict convolution
    (correlation = convolution with a pre-flipped kernel).  For symmetric
    kernels (mean, Gaussian, Laplacian) the two operations are identical.
    This implementation also uses correlation for consistency; the kernel is
    NOT flipped.  To perform strict convolution, flip the kernel before
    calling: ``kernel[::-1, ::-1]``.

    Algorithm
    ---------
    Uses NumPy stride tricks (``as_strided``) to build a view of the image
    that groups each (kH, kW) neighbourhood without copying data, then
    contracts it with the kernel via ``einsum``.  This is *vectorised*:
    no Python loop over rows or columns.

    Complexity: O(H · W · kH · kW)  — same as any direct convolution.

    Parameters
    ----------
    img      : np.ndarray – 2-D float64 image, shape (H, W).
    kernel   : np.ndarray – 2-D numeric kernel, shape (kH, kW), odd dimensions.
    pad_mode : str        – padding mode passed to :func:`pad` (default 'constant').

    Returns
    -------
    np.ndarray float64, shape (H, W) — same spatial size as input.

    Raises
    ------
    TypeError  – img or kernel wrong type.
    ValueError – img is not 2-D, kernel has even dimension, etc.
    """
    _validate_image(img, ndim=2)
    _validate_kernel(kernel)

    img = img.astype(np.float64)
    k = kernel.astype(np.float64)

    kH, kW = k.shape
    pH, pW = kH // 2, kW // 2

    # 1. Pad the image so output has the same size as input
    padded = pad(img, pH, pW, mode=pad_mode)

    H, W = img.shape
    pHt, pWt = padded.shape

    # 2. Build a sliding-window view using stride tricks (no data copy)
    #    Shape: (H, W, kH, kW)
    from numpy.lib.stride_tricks import as_strided
    shape = (H, W, kH, kW)
    strides = (
        padded.strides[0],  # step one output row   → one row in padded
        padded.strides[1],  # step one output col   → one col in padded
        padded.strides[0],  # step one kernel row   → one row in padded
        padded.strides[1],  # step one kernel col   → one col in padded
    )
    windows = as_strided(padded, shape=shape, strides=strides)

    # 3. Element-wise multiply each window by the kernel and sum
    #    einsum 'ijkl,kl->ij' : contract (kH, kW) dimensions
    result = np.einsum("ijkl,kl->ij", windows, k)
    return result


# ---------------------------------------------------------------------------
# Spatial filtering dispatcher (grayscale + RGB)
# ---------------------------------------------------------------------------

def spatial_filter(
    img: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Apply *kernel* to a **grayscale OR RGB** image.

    Strategy for RGB
    ~~~~~~~~~~~~~~~~
    The kernel is applied independently to each colour channel (R, G, B).
    This is valid for linear, shift-invariant filters (mean, Gaussian, Sobel)
    because each channel can be treated as an independent 2-D signal.
    Non-linear filters (median) should NOT use this function; see
    :func:`minicv.filters.median_filter`.

    Parameters
    ----------
    img      : np.ndarray – 2-D (H, W) float64 grayscale, or 3-D (H, W, 3) float64 RGB.
    kernel   : np.ndarray – 2-D numeric kernel with odd dimensions.
    pad_mode : str        – border handling: 'constant', 'reflect', 'replicate'.

    Returns
    -------
    np.ndarray float64, same shape as *img*.

    Raises
    ------
    ValueError – img is not 2-D or 3-D.
    """
    _validate_image(img)
    _validate_kernel(kernel)
    
    if img.ndim == 2:
        return convolve2d(img, kernel, pad_mode=pad_mode)
    img = to_float64(img)
    if img.ndim == 3:
        channels = [
            convolve2d(img[:, :, c], kernel, pad_mode=pad_mode)
            for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=2)

    raise ValueError(
        f"Image must be 2-D (grayscale) or 3-D (RGB), got {img.ndim}-D."
    )
