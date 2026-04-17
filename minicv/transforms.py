"""
minicv.transforms
=================
Geometric image transformations implemented from scratch with NumPy.

All transformations use an **inverse mapping** strategy:
    For each destination pixel (x', y'), compute the source coordinate
    (x, y) and sample from the input image.

This avoids holes in the output image that forward mapping can cause.

Contents
--------
* resize    – scale image to new (height, width) with nearest or bilinear interp.
* rotate    – rotate about image centre with configurable interpolation.
* translate – shift image by (tx, ty) pixels.
"""

import numpy as np
from .utils import _validate_image, pad, to_float64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """
    Sample *img* at sub-pixel coordinates using bilinear interpolation.

    For each (y, x) pair, the output is the weighted average of the four
    surrounding integer-coordinate pixels:

        f(y,x) ≈ (1-dy)(1-dx)·f(y0,x0)
                + (1-dy)·dx  ·f(y0,x1)
                +    dy·(1-dx)·f(y1,x0)
                +    dy·dx   ·f(y1,x1)

    Coordinates outside [0, H-1] × [0, W-1] are clamped to the border.

    Parameters
    ----------
    img : np.ndarray – source image, 2-D (H, W) or 3-D (H, W, C), float64.
    yy  : np.ndarray – 2-D array of y (row) coordinates.
    xx  : np.ndarray – 2-D array of x (col) coordinates.

    Returns
    -------
    np.ndarray float64, shape (yy.shape[0], yy.shape[1][, C]).
    """
    H, W = img.shape[:2]

    # Integer parts
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional parts
    dy = yy - y0
    dx = xx - x0

    # Clamp to valid range
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)

    if img.ndim == 2:
        return (
            (1 - dy) * (1 - dx) * img[y0, x0]
            + (1 - dy) * dx * img[y0, x1]
            + dy * (1 - dx) * img[y1, x0]
            + dy * dx * img[y1, x1]
        )
    else:
        # Expand dims for broadcasting over channels
        dy = dy[:, :, np.newaxis]
        dx = dx[:, :, np.newaxis]
        return (
            (1 - dy) * (1 - dx) * img[y0, x0]
            + (1 - dy) * dx * img[y0, x1]
            + dy * (1 - dx) * img[y1, x0]
            + dy * dx * img[y1, x1]
        )


def _nearest_sample(img: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """
    Sample *img* at coordinates (yy, xx) using nearest-neighbour interpolation.

    Rounds each coordinate to the closest integer then clamps to image bounds.

    Parameters
    ----------
    img : np.ndarray – source image, 2-D or 3-D, float64.
    yy  : np.ndarray – 2-D array of y coordinates.
    xx  : np.ndarray – 2-D array of x coordinates.

    Returns
    -------
    np.ndarray float64, same channel depth as *img*.
    """
    H, W = img.shape[:2]
    yr = np.clip(np.round(yy).astype(int), 0, H - 1)
    xr = np.clip(np.round(xx).astype(int), 0, W - 1)
    return img[yr, xr].astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resize(
    img: np.ndarray,
    new_h: int,
    new_w: int,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize *img* to (new_h, new_w) using inverse mapping.

    Interpolation methods
    ---------------------
    nearest
        Each output pixel samples the nearest input pixel.
        Fast; can produce blocky artefacts when upscaling.

    bilinear
        Each output pixel is a weighted average of the 4 nearest input pixels.
        Smoother result; slight blurring, especially when downscaling.

    Inverse mapping formula
    -----------------------
    Scale factors:  sy = H_src / H_dst,  sx = W_src / W_dst
    For destination pixel (r, c):
        source_y = (r + 0.5) * sy − 0.5
        source_x = (c + 0.5) * sx − 0.5

    The ±0.5 correction aligns pixel centres between source and destination
    grids (same as OpenCV's INTER_LINEAR).

    Parameters
    ----------
    img           : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    new_h, new_w  : int        – output height and width (> 0).
    interpolation : str        – 'nearest' or 'bilinear'.

    Returns
    -------
    np.ndarray float64, shape (new_h, new_w[, C]).

    Raises
    ------
    ValueError – new_h or new_w ≤ 0, or unknown interpolation.
    """
    _validate_image(img)
    img=to_float64(img)
    if new_h <= 0 or new_w <= 0:
        raise ValueError(
            f"new_h and new_w must be > 0, got ({new_h}, {new_w})."
        )
    valid = {"nearest", "bilinear"}
    if interpolation not in valid:
        raise ValueError(f"'interpolation' must be one of {valid}, got '{interpolation}'.")

    H, W = img.shape[:2]
    sy = H / new_h
    sx = W / new_w

    # Build destination coordinate grids
    rows = np.arange(new_h, dtype=np.float64)
    cols = np.arange(new_w, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)  # cc[r,c] = c,  rr[r,c] = r

    # Map destination → source (inverse mapping)
    src_y = (rr + 0.5) * sy - 0.5
    src_x = (cc + 0.5) * sx - 0.5

    sampler = _bilinear_sample if interpolation == "bilinear" else _nearest_sample
    return sampler(img.astype(np.float64), src_y, src_x)


def rotate(
    img: np.ndarray,
    angle_deg: float,
    interpolation: str = "bilinear",
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Rotate *img* about its centre by *angle_deg* degrees counter-clockwise.

    Inverse mapping formula
    -----------------------
    Given a destination pixel (x', y') in normalised (shifted) coordinates,
    the corresponding source pixel is obtained by applying the inverse rotation:

        [x]   [ cos θ   sin θ] [x' − cx]   [cx]
        [y] = [-sin θ   cos θ] [y' − cy] + [cy]

    where (cx, cy) is the image centre and θ = angle in radians.

    Parameters
    ----------
    img          : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    angle_deg    : float      – counter-clockwise rotation angle in degrees.
    interpolation: str        – 'nearest' or 'bilinear'.
    fill         : float      – background fill value for pixels outside source bounds.

    Returns
    -------
    np.ndarray float64, same shape as *img*.

    Notes
    -----
    The output image has the same spatial dimensions as the input.
    Pixels that map outside the source are filled with *fill*.
    """
    _validate_image(img)
    img=to_float64(img)
    valid = {"nearest", "bilinear"}
    if interpolation not in valid:
        raise ValueError(f"'interpolation' must be one of {valid}.")

    H, W = img.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Destination grid
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)

    # Shift to centre
    xp = cc - cx
    yp = rr - cy

    # Inverse rotation
    src_x = cos_t * xp + sin_t * yp + cx
    src_y = -sin_t * xp + cos_t * yp + cy

    # Mask pixels that fall outside the source image
    valid_mask = (
        (src_y >= 0) & (src_y <= H - 1)
        & (src_x >= 0) & (src_x <= W - 1)
    )

    src = img.astype(np.float64)
    sampler = _bilinear_sample if interpolation == "bilinear" else _nearest_sample
    out = sampler(src, src_y, src_x)

    # Fill out-of-bounds regions
    if img.ndim == 2:
        out[~valid_mask] = fill
    else:
        out[~valid_mask] = fill

    return out


def translate(
    img: np.ndarray,
    tx: float,
    ty: float,
    interpolation: str = "bilinear",
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Shift *img* by (tx, ty) pixels.

    Convention: tx > 0 shifts right; ty > 0 shifts down.

    Inverse mapping:
        src_x = dst_x − tx
        src_y = dst_y − ty

    Parameters
    ----------
    img           : np.ndarray – float64 grayscale (H, W) or RGB (H, W, 3).
    tx            : float      – horizontal shift in pixels (right = positive).
    ty            : float      – vertical shift in pixels (down = positive).
    interpolation : str        – 'nearest' or 'bilinear'.
    fill          : float      – value for pixels shifted in from outside.

    Returns
    -------
    np.ndarray float64, same shape as *img*.
    """
    _validate_image(img)
    img=to_float64(img)
    valid = {"nearest", "bilinear"}
    if interpolation not in valid:
        raise ValueError(f"'interpolation' must be one of {valid}.")

    H, W = img.shape[:2]
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)

    src_x = cc - tx
    src_y = rr - ty

    valid_mask = (
        (src_y >= 0) & (src_y <= H - 1)
        & (src_x >= 0) & (src_x <= W - 1)
    )

    src = img.astype(np.float64)
    sampler = _bilinear_sample if interpolation == "bilinear" else _nearest_sample
    out = sampler(src, src_y, src_x)

    if img.ndim == 2:
        out[~valid_mask] = fill
    else:
        out[~valid_mask] = fill

    return out
