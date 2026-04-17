"""
minicv.filters
==============
Spatial filtering and image-processing operations.

All functions operate on **float64** arrays in [0, 1] (use
:func:`minicv.utils.to_float64` before calling).

Contents
--------
Filtering
~~~~~~~~~
* mean_filter          – box / averaging filter
* gaussian_kernel      – generate a Gaussian kernel
* gaussian_filter      – Gaussian smoothing via convolution
* median_filter        – non-linear rank filter (vectorised sliding window)

Thresholding
~~~~~~~~~~~~
* threshold_global     – single-value binarisation
* threshold_otsu       – automatic threshold by inter-class variance
* threshold_adaptive   – local neighbourhood thresholding

Edge Detection
~~~~~~~~~~~~~~
* sobel_gradients      – Gx, Gy, magnitude, angle

Bit-plane & Histogram
~~~~~~~~~~~~~~~~~~~~~
* bit_plane_slice      – extract single bit-plane from uint8 image
* histogram            – compute normalised or count histogram
* equalize_histogram   – histogram equalisation (grayscale)

Additional Techniques
~~~~~~~~~~~~~~~~~~~~~
* laplacian_filter     – second-order edge detection
* unsharp_mask         – high-frequency sharpening
"""

import numpy as np
from .utils import (
    _validate_image,
    _validate_kernel,
    convolve2d,
    spatial_filter,
    pad,
)


# ===========================================================================
# 4.1  Mean / Box filter
# ===========================================================================

def mean_filter(
    img: np.ndarray,
    kernel_size: int = 3,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Apply a mean (box) filter to *img*.

    Every output pixel is the arithmetic mean of the (kernel_size × kernel_size)
    neighbourhood centred on that pixel.  This is equivalent to convolving with
    a kernel whose entries all equal 1 / (kernel_size²).

    Parameters
    ----------
    img         : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    kernel_size : int        – side length of the square kernel; must be odd ≥ 1.
    pad_mode    : str        – 'constant', 'reflect', or 'replicate'.

    Returns
    -------
    np.ndarray float64, same shape as *img*, values roughly in [0, 1].

    Raises
    ------
    ValueError – kernel_size is even or < 1.

    Notes
    -----
    Mean filtering is a **low-pass** filter: it attenuates high-frequency
    detail (edges, noise) while preserving low-frequency content (broad
    shapes).  It is separable (H·V = 2-D box), so the stride-trick
    convolution in :func:`utils.convolve2d` is efficient.
    """
    _validate_image(img)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"'kernel_size' must be an odd integer ≥ 1, got {kernel_size}."
        )
    k = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    return spatial_filter(img, k, pad_mode=pad_mode)


# ===========================================================================
# 4.2  Gaussian filter
# ===========================================================================

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a normalised 2-D Gaussian kernel.

    The kernel is computed by sampling the Gaussian function on a grid
    centred at the origin and then normalising so entries sum to 1.

    Formula
    -------
    G(x, y) = exp(−(x² + y²) / (2 σ²))

    After sampling: K = G / sum(G)

    Parameters
    ----------
    size  : int   – side length of the square kernel (must be odd ≥ 1).
    sigma : float – standard deviation of the Gaussian (must be > 0).

    Returns
    -------
    np.ndarray float64, shape (size, size), entries summing to 1.

    Raises
    ------
    ValueError – size is even or sigma ≤ 0.

    Notes
    -----
    OpenCV uses size=0 with an explicit sigma to auto-compute size via
    ``size = ceil(3*sigma) | 1 * 2 + 1``.  We require the caller to be
    explicit for clarity.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"'size' must be odd and ≥ 1, got {size}.")
    if sigma <= 0:
        raise ValueError(f"'sigma' must be > 0, got {sigma}.")

    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    # Outer product gives 2-D squared distances
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_filter(
    img: np.ndarray,
    size: int = 5,
    sigma: float = 1.0,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Smooth *img* with a Gaussian kernel.

    Parameters
    ----------
    img      : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    size     : int        – kernel side length (odd, ≥ 1).
    sigma    : float      – Gaussian standard deviation (> 0).
    pad_mode : str        – border handling mode.

    Returns
    -------
    np.ndarray float64, same shape as *img*.

    Notes
    -----
    Gaussian filtering is the canonical image smoothing method.  Unlike mean
    filtering it weights closer pixels more heavily, yielding fewer blocking
    artefacts.  It is also separable (2-D Gaussian = 1-D ⊗ 1-D), but we
    apply the full 2-D kernel for simplicity and correctness.
    """
    _validate_image(img)
    k = gaussian_kernel(size, sigma)
    return spatial_filter(img, k, pad_mode=pad_mode)


# ===========================================================================
# 4.3  Median filter
# ===========================================================================

def median_filter(
    img: np.ndarray,
    kernel_size: int = 3,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply a median filter to *img*.

    The output pixel equals the **median** of the (kernel_size × kernel_size)
    neighbourhood.  Median filtering is excellent at removing **salt-and-pepper
    noise** while preserving edges far better than mean or Gaussian filtering.

    Why loops are necessary
    ~~~~~~~~~~~~~~~~~~~~~~~
    The median is a non-linear operation: it cannot be expressed as a dot
    product with a fixed kernel.  True vectorisation therefore requires
    sorting each neighbourhood independently.  We use NumPy ``as_strided``
    to gather all windows into a 4-D array (H, W, kH, kW) in one shot (no
    copy), then call ``np.median`` along the last two axes.  This avoids any
    explicit Python loop over individual pixels; the only "iteration" is
    internal to NumPy's C-level sort.

    Parameters
    ----------
    img         : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    kernel_size : int        – odd integer ≥ 1.
    pad_mode    : str        – border handling.

    Returns
    -------
    np.ndarray float64, same shape as *img*.

    Raises
    ------
    ValueError – kernel_size is even or < 1.
    """
    _validate_image(img)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"'kernel_size' must be an odd integer ≥ 1, got {kernel_size}."
        )

    def _median_gray(channel: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided
        pH = pW = kernel_size // 2
        padded = pad(channel, pH, pW, mode=pad_mode)
        H, W = channel.shape
        kS = kernel_size
        shape = (H, W, kS, kS)
        strides = (
            padded.strides[0],
            padded.strides[1],
            padded.strides[0],
            padded.strides[1],
        )
        windows = as_strided(padded, shape=shape, strides=strides)
        # Reshape to (H, W, kS*kS) and take median along last axis
        return np.median(windows.reshape(H, W, -1), axis=2)

    if img.ndim == 2:
        return _median_gray(img.astype(np.float64))
    # RGB: apply per-channel
    channels = [
        _median_gray(img[:, :, c].astype(np.float64))
        for c in range(img.shape[2])
    ]
    return np.stack(channels, axis=2)


# ===========================================================================
# 4.4  Thresholding
# ===========================================================================

def threshold_global(
    img: np.ndarray,
    thresh: float,
    *,
    max_val: float = 1.0,
) -> np.ndarray:
    """
    Binarise a grayscale image using a single fixed threshold.

    Rule:  output[i,j] = max_val  if img[i,j] > thresh  else 0

    Parameters
    ----------
    img      : np.ndarray – 2-D float64 grayscale in [0, 1].
    thresh   : float      – decision boundary; pixels above → max_val.
    max_val  : float      – value assigned to foreground pixels (default 1.0).

    Returns
    -------
    np.ndarray float64, shape (H, W), values in {0, max_val}.

    Raises
    ------
    ValueError – img is not 2-D or thresh is outside [0, 1].
    """
    _validate_image(img, ndim=2)
    if not (0.0 <= thresh <= 1.0):
        raise ValueError(f"'thresh' must be in [0, 1], got {thresh}.")
    return np.where(img > thresh, max_val, 0.0)


def threshold_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Automatically determine an optimal threshold via Otsu's method and
    return both the binary image and the threshold value.

    Algorithm
    ---------
    Otsu (1979) maximises the **between-class variance** σ²_B over all
    possible thresholds t ∈ [0, 255] (computed on the integer histogram):

        σ²_B(t) = ω₀(t) · ω₁(t) · [μ₀(t) − μ₁(t)]²

    where ω₀, ω₁ are the class probabilities and μ₀, μ₁ are the class means.
    The threshold that maximises σ²_B is chosen.

    Parameters
    ----------
    img : np.ndarray – 2-D float64 grayscale in [0, 1].

    Returns
    -------
    (binary_image, threshold)
        binary_image : np.ndarray float64, shape (H, W), values in {0, 1}.
        threshold    : float in [0, 1] – chosen threshold.

    Raises
    ------
    ValueError – img is not 2-D.
    """
    _validate_image(img, ndim=2)

    # Work in 8-bit integer space for the histogram (256 levels)
    img8 = (img * 255).astype(np.uint8).astype(np.float64)
    hist, _ = np.histogram(img8, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = img8.size

    # Cumulative sums and means from left (background side)
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256, dtype=np.float64))

    global_mean = cum_mean[-1]
    p_bg = cum_sum / total                             # ω₀
    p_fg = 1.0 - p_bg                                 # ω₁

    # Avoid division by zero at the extremes
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(p_bg > 0, cum_mean / cum_sum, 0.0)
        mean_fg = np.where(
            p_fg > 0,
            (global_mean - cum_mean) / (total - cum_sum + 1e-10),
            0.0,
        )
        sigma_b2 = p_bg * p_fg * (mean_bg - mean_fg) ** 2

    best_t = int(np.argmax(sigma_b2))
    threshold = best_t / 255.0
    binary = (img > threshold).astype(np.float64)
    return binary, threshold


def threshold_adaptive(
    img: np.ndarray,
    block_size: int = 11,
    C: float = 0.02,
    method: str = "mean",
) -> np.ndarray:
    """
    Binarise *img* using a locally computed threshold for each pixel.

    Useful when illumination is non-uniform: a single global threshold fails
    in shadowed regions, but a local threshold adapts to the local brightness.

    Algorithm
    ---------
    For each pixel (i, j):
        local_thresh = mean(neighbourhood) − C        (method='mean')
        local_thresh = gaussian_weighted_mean(neighbourhood) − C  (method='gaussian')
        output[i,j]  = 1 if img[i,j] > local_thresh else 0

    Parameters
    ----------
    img        : np.ndarray – 2-D float64 grayscale in [0, 1].
    block_size : int        – side of the square neighbourhood (odd, ≥ 3).
    C          : float      – constant subtracted from the local mean to fine-
                              tune the threshold (positive → more background).
    method     : str        – 'mean' or 'gaussian'.

    Returns
    -------
    np.ndarray float64, shape (H, W), values in {0, 1}.

    Raises
    ------
    ValueError – block_size is even, img is not 2-D, or method unknown.
    """
    _validate_image(img, ndim=2)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"'block_size' must be an odd integer ≥ 3, got {block_size}."
        )
    valid_methods = {"mean", "gaussian"}
    if method not in valid_methods:
        raise ValueError(f"'method' must be one of {valid_methods}, got '{method}'.")

    if method == "mean":
        k = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
    else:  # gaussian
        from .utils import to_float64
        k = gaussian_kernel(block_size, sigma=block_size / 6.0)

    local_mean = convolve2d(img.astype(np.float64), k, pad_mode="reflect")
    binary = (img > (local_mean - C)).astype(np.float64)
    return binary


# ===========================================================================
# 4.5  Sobel gradients
# ===========================================================================

def sobel_gradients(
    img: np.ndarray,
    pad_mode: str = "reflect",
) -> dict:
    """
    Compute Sobel gradient images from a grayscale image.

    The Sobel operator approximates the first derivative using the kernels:

        Kx = [[-1,  0,  1],        Ky = [[-1, -2, -1],
              [-2,  0,  2],               [ 0,  0,  0],
              [-1,  0,  1]]               [ 1,  2,  1]]

    Kx detects vertical edges (horizontal gradient).
    Ky detects horizontal edges (vertical gradient).

    Derived quantities
    ------------------
    magnitude = sqrt(Gx² + Gy²)
    angle     = arctan2(Gy, Gx)  [radians]

    Parameters
    ----------
    img      : np.ndarray – 2-D float64 grayscale.
    pad_mode : str        – border handling.

    Returns
    -------
    dict with keys:
        'Gx'        – horizontal gradient, float64 (H, W)
        'Gy'        – vertical gradient, float64 (H, W)
        'magnitude' – gradient magnitude, float64 (H, W), values ≥ 0
        'angle'     – gradient direction in radians, float64 (H, W)
    """
    _validate_image(img, ndim=2)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    Gx = convolve2d(img.astype(np.float64), Kx, pad_mode=pad_mode)
    Gy = convolve2d(img.astype(np.float64), Ky, pad_mode=pad_mode)
    return {
        "Gx": Gx,
        "Gy": Gy,
        "magnitude": np.sqrt(Gx ** 2 + Gy ** 2),
        "angle": np.arctan2(Gy, Gx),
    }


# ===========================================================================
# 4.6  Bit-plane slicing
# ===========================================================================

def bit_plane_slice(img: np.ndarray, plane: int) -> np.ndarray:
    """
    Extract a single bit-plane from a grayscale image.

    Each 8-bit pixel is an integer 0–255.  Bit-plane *n* (0 = LSB, 7 = MSB)
    is the image whose pixels are the *n*-th bit of each original pixel.

    Formula: plane_image[i,j] = (uint8_pixel[i,j] >> plane) & 1

    The MSB plane (plane=7) carries the most structural information about
    the image (shapes, large regions); the LSB (plane=0) is close to random
    noise for natural images.

    Parameters
    ----------
    img   : np.ndarray – 2-D grayscale, any numeric dtype.
    plane : int        – bit index in [0, 7] (0 = LSB, 7 = MSB).

    Returns
    -------
    np.ndarray float64, shape (H, W), values in {0, 1}.

    Raises
    ------
    ValueError – plane is outside [0, 7] or img is not 2-D.
    """
    _validate_image(img, ndim=2)
    if not (0 <= plane <= 7):
        raise ValueError(f"'plane' must be in [0, 7], got {plane}.")
    img8 = (np.clip(img.astype(np.float64), 0, 1) * 255).astype(np.uint8)
    return ((img8 >> plane) & 1).astype(np.float64)


# ===========================================================================
# 4.7  Histogram + histogram equalisation
# ===========================================================================

def histogram(
    img: np.ndarray,
    bins: int = 256,

) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pixel-value histogram of a grayscale image.

    Parameters
    ----------
    img        : np.ndarray – 2-D float64 grayscale, values in [0, 1].
    bins       : int        – number of histogram bins (default 256).

    Returns
    -------
    (counts, bin_edges)
        counts    : np.ndarray float64, shape (bins,)
        bin_edges : np.ndarray float64, shape (bins+1,)

    Raises
    ------
    ValueError – img is not 2-D.
    """
    _validate_image(img, ndim=2)
    counts, edges = np.histogram(
        img.ravel(), bins=bins, range=(0.0, 1.0)
    )
    return counts.astype(np.float64), edges


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast of a grayscale image via histogram equalisation.

    Algorithm
    ---------
    1. Compute normalised histogram  p(k)   for k = 0 … 255.
    2. Compute CDF:  cdf(k) = Σ_{j≤k} p(j).
    3. Map each pixel:  out = round(cdf(pixel) * 255) / 255.

    This remaps pixel values so the output histogram is approximately flat,
    spreading the most common intensities across the full range.

    Parameters
    ----------
    img : np.ndarray – 2-D float64 grayscale in [0, 1].

    Returns
    -------
    np.ndarray float64, shape (H, W), values in [0, 1].

    Raises
    ------
    ValueError – img is not 2-D.
    """
    _validate_image(img, ndim=2)
    img8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    hist, _ = np.histogram(img8.ravel(), bins=256, range=(0, 255))
    cdf = hist.cumsum().astype(np.float64)
    # Normalise CDF to [0, 1]
    cdf /= cdf[-1]
    # Apply mapping using the pixel value as LUT index
    out8 = cdf[img8]
    return out8


# ===========================================================================
# 4.8  Additional techniques: Laplacian & Unsharp Masking
# ===========================================================================

def laplacian_filter(
    img: np.ndarray,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Detect edges using the Laplacian (second-order derivative) operator.

    Kernel
    ------
    L = [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]

    This is the discrete Laplacian ∇²f, which highlights regions of rapid
    intensity change (edges) in all directions simultaneously.  Unlike Sobel
    it is isotropic and directionally neutral.

    Parameters
    ----------
    img      : np.ndarray – 2-D float64 grayscale.
    pad_mode : str        – border handling.

    Returns
    -------
    np.ndarray float64, shape (H, W) – Laplacian response (can be negative).
    """
    _validate_image(img, ndim=2)
    K = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    return convolve2d(img.astype(np.float64), K, pad_mode=pad_mode)


def unsharp_mask(
    img: np.ndarray,
    sigma: float = 2.0,
    strength: float = 1.5,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Sharpen *img* using unsharp masking.

    Algorithm
    ---------
    blurred   = Gaussian(img, sigma)
    detail    = img − blurred          (high-frequency mask)
    output    = img + strength * detail

    Unsharp masking amplifies high-frequency content (fine edges, textures)
    by adding back a scaled residual between the original and its blurred
    version.  Despite the name it sharpens rather than blurs.

    Parameters
    ----------
    img      : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    sigma    : float      – Gaussian standard deviation for the blur step.
    strength : float      – how much of the detail is added back (> 0).
    pad_mode : str        – border handling for the Gaussian step.

    Returns
    -------
    np.ndarray float64, same shape as *img*, clipped to [0, 1].
    """
    _validate_image(img)
    size = int(np.ceil(3 * sigma) * 2 + 1) | 1   # auto-size, always odd
    if size < 3:
        size = 3
    blurred = gaussian_filter(img.astype(np.float64), size=size, sigma=sigma,
                              pad_mode=pad_mode)
    detail = img.astype(np.float64) - blurred
    return np.clip(img.astype(np.float64) + strength * detail, 0.0, 1.0)
