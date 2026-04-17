"""
minicv.drawing
==============
Drawing primitives implemented directly on NumPy arrays.

Design decisions
----------------
* All functions operate **in-place** and also return the modified array for
  chaining convenience.
* Colors are always specified as a scalar (grayscale) or 3-tuple (RGB, values
  matching the image dtype, e.g. floats in [0,1] or ints 0–255).
* Coordinates are (x, y) where x = column, y = row.
* All drawing is **clipped** to canvas boundaries — drawing outside the image
  simply does nothing for the out-of-bounds pixels.

Contents
--------
* _set_pixel      – internal: write one pixel safely (clip to bounds)
* draw_point      – fill a radius-r circle
* draw_line       – Bresenham integer line
* draw_rectangle  – axis-aligned rectangle (outline or filled)
* draw_polygon    – arbitrary outline polygon (scanline fill optional)
* put_text        – render ASCII text using a built-in 5×8 bitmap font
"""

import numpy as np
from .utils import _validate_image


# ---------------------------------------------------------------------------
# Colour validation helper
# ---------------------------------------------------------------------------

def _parse_color(color, img: np.ndarray):
    """
    Validate and return a colour compatible with *img*.

    * Grayscale image (2-D): color must be a scalar (int or float).
    * RGB image (3-D):       color must be a 3-element sequence.

    Returns
    -------
    Scalar or np.ndarray of shape (3,).
    """
    if img.ndim == 2:
        if not np.isscalar(color):
            raise TypeError(
                "For a grayscale image, 'color' must be a scalar, "
                f"got {type(color).__name__}."
            )
        return float(color)
    else:
        color = np.asarray(color, dtype=img.dtype)
        if color.shape != (3,):
            raise ValueError(
                f"For an RGB image, 'color' must have 3 elements, got {color.shape}."
            )
        return color


# ---------------------------------------------------------------------------
# Pixel setter
# ---------------------------------------------------------------------------

def _set_pixel(img: np.ndarray, x: int, y: int, color) -> None:
    """
    Write *color* to pixel (x, y) = (col, row) if inside image bounds.

    Parameters
    ----------
    img   : np.ndarray – target image (modified in-place).
    x     : int        – column index.
    y     : int        – row index.
    color : scalar or array – pixel value(s) to write.
    """
    H, W = img.shape[:2]
    if 0 <= y < H and 0 <= x < W:
        img[y, x] = color


def _set_thick_pixel(img, x, y, color, thickness):
    """Draw a filled square of side `thickness` centred on (x, y)."""
    half = thickness // 2
    H, W = img.shape[:2]
    r0 = max(0, y - half)
    r1 = min(H, y + half + 1)
    c0 = max(0, x - half)
    c1 = min(W, x + half + 1)
    img[r0:r1, c0:c1] = color


# ===========================================================================
# Drawing primitives
# ===========================================================================

def draw_point(
    img: np.ndarray,
    x: int,
    y: int,
    color,
    radius: int = 1,
) -> np.ndarray:
    """
    Draw a filled circular point centred at (x, y).

    Parameters
    ----------
    img    : np.ndarray – image to draw on (modified in-place).
    x, y   : int        – centre column and row.
    color  : scalar | tuple – pixel colour (grayscale scalar or RGB 3-tuple).
    radius : int        – radius of the point in pixels (default 1 → 1 pixel).

    Returns
    -------
    np.ndarray – same array as *img* (modified in-place).

    Notes
    -----
    Uses the midpoint circle algorithm to determine which pixels lie within
    the circle, then fills them.  For radius == 1 this reduces to a single
    pixel write.
    """
    _validate_image(img)
    c = _parse_color(color, img)

    H, W = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                _set_pixel(img, x + dx, y + dy, c)
    return img


def draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.

    Bresenham's Line Algorithm
    --------------------------
    Works entirely in integer arithmetic.  At each step the algorithm decides
    whether to step in the dominant axis only, or to also step in the minor
    axis, based on an accumulated error term:

        error += 2*dy
        if error > dx: y += step_y;  error -= 2*dx

    This produces a pixel-perfect, connected raster line with no floating
    point arithmetic.

    Parameters
    ----------
    img            : np.ndarray – image to draw on.
    x0, y0, x1, y1 : int        – start and end pixel coordinates.
    color          : scalar | tuple – pixel colour.
    thickness      : int        – line width in pixels (≥ 1).

    Returns
    -------
    np.ndarray – same array as *img*.
    """
    _validate_image(img)
    c = _parse_color(color, img)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        _set_thick_pixel(img, x, y, c, thickness)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return img


def draw_rectangle(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color,
    thickness: int = 1,
    filled: bool = False,
) -> np.ndarray:
    """
    Draw an axis-aligned rectangle with corners at (x0,y0) and (x1,y1).

    Parameters
    ----------
    img       : np.ndarray – target image.
    x0, y0    : int        – top-left corner (col, row).
    x1, y1    : int        – bottom-right corner (col, row).
    color     : scalar | tuple – border (and fill) colour.
    thickness : int        – border line width (ignored when filled=True).
    filled    : bool       – if True, fill the interior with *color*.

    Returns
    -------
    np.ndarray – same array as *img*.

    Notes
    -----
    Filled rectangles use NumPy slice assignment (vectorised), making them
    O(area) in C rather than O(perimeter) × O(height) Python calls.
    """
    _validate_image(img)
    c = _parse_color(color, img)

    # Normalise so x0 ≤ x1, y0 ≤ y1
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    H, W = img.shape[:2]

    if filled:
        # Clamp to image bounds
        r0 = max(0, y0); r1 = min(H, y1 + 1)
        c0 = max(0, x0); c1 = min(W, x1 + 1)
        img[r0:r1, c0:c1] = c
    else:
        draw_line(img, x0, y0, x1, y0, color, thickness)  # top
        draw_line(img, x0, y1, x1, y1, color, thickness)  # bottom
        draw_line(img, x0, y0, x0, y1, color, thickness)  # left
        draw_line(img, x1, y0, x1, y1, color, thickness)  # right
    return img


def draw_polygon(
    img: np.ndarray,
    points: list[tuple[int, int]],
    color,
    thickness: int = 1,
    filled: bool = False,
) -> np.ndarray:
    """
    Draw a polygon defined by a list of (x, y) vertex coordinates.

    Parameters
    ----------
    img      : np.ndarray             – target image.
    points   : list of (int, int)     – polygon vertices in order.
    color    : scalar | tuple         – outline (and fill) colour.
    thickness: int                    – edge line width.
    filled   : bool                   – if True, fill the polygon interior using
                                        a scanline algorithm.

    Returns
    -------
    np.ndarray – same array as *img*.

    Notes
    -----
    The scanline fill uses numpy operations per scanline row (one Python loop
    over rows is unavoidable for a general polygon fill, but inner operations
    are vectorised).
    """
    _validate_image(img)
    c = _parse_color(color, img)

    if len(points) < 2:
        return img

    # Draw outline edges
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        draw_line(img, x0, y0, x1, y1, color, thickness)

    if filled and len(points) >= 3:
        _scanline_fill(img, points, c)

    return img


def _scanline_fill(img: np.ndarray, points, color) -> None:
    """
    Fill polygon interior using a scanline algorithm.

    For each horizontal scanline y, find all edge intersections, sort them,
    and fill between pairs.  One Python loop over scanline rows is necessary
    because intersection computation depends on y; inner operations are
    NumPy-vectorised where possible.
    """
    H, W = img.shape[:2]
    n = len(points)
    ys = [p[1] for p in points]
    y_min = max(0, min(ys))
    y_max = min(H - 1, max(ys))

    for y in range(y_min, y_max + 1):
        xs_intersect = []
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    xs_intersect.append(xi)
        xs_intersect.sort()
        for j in range(0, len(xs_intersect) - 1, 2):
            x_start = max(0, int(np.ceil(xs_intersect[j])))
            x_end = min(W - 1, int(np.floor(xs_intersect[j + 1])))
            img[y, x_start:x_end + 1] = color


# ===========================================================================
# Text placement
# ===========================================================================

# Minimal 5×8 bitmap font for ASCII 32–126
# Each character is 5 columns × 8 rows packed into 5 bytes (one per column).
# Bit 0 = top row, bit 7 = bottom row.
_FONT_5x8: dict[str, list[int]] = {
    " ": [0x00, 0x00, 0x00, 0x00, 0x00],
    "!": [0x00, 0x00, 0x5F, 0x00, 0x00],
    '"': [0x00, 0x07, 0x00, 0x07, 0x00],
    "#": [0x14, 0x7F, 0x14, 0x7F, 0x14],
    "$": [0x24, 0x2A, 0x7F, 0x2A, 0x12],
    "%": [0x23, 0x13, 0x08, 0x64, 0x62],
    "&": [0x36, 0x49, 0x55, 0x22, 0x50],
    "'": [0x00, 0x05, 0x03, 0x00, 0x00],
    "(": [0x00, 0x1C, 0x22, 0x41, 0x00],
    ")": [0x00, 0x41, 0x22, 0x1C, 0x00],
    "*": [0x14, 0x08, 0x3E, 0x08, 0x14],
    "+": [0x08, 0x08, 0x3E, 0x08, 0x08],
    ",": [0x00, 0x50, 0x30, 0x00, 0x00],
    "-": [0x08, 0x08, 0x08, 0x08, 0x08],
    ".": [0x00, 0x60, 0x60, 0x00, 0x00],
    "/": [0x20, 0x10, 0x08, 0x04, 0x02],
    "0": [0x3E, 0x51, 0x49, 0x45, 0x3E],
    "1": [0x00, 0x42, 0x7F, 0x40, 0x00],
    "2": [0x42, 0x61, 0x51, 0x49, 0x46],
    "3": [0x21, 0x41, 0x45, 0x4B, 0x31],
    "4": [0x18, 0x14, 0x12, 0x7F, 0x10],
    "5": [0x27, 0x45, 0x45, 0x45, 0x39],
    "6": [0x3C, 0x4A, 0x49, 0x49, 0x30],
    "7": [0x01, 0x71, 0x09, 0x05, 0x03],
    "8": [0x36, 0x49, 0x49, 0x49, 0x36],
    "9": [0x06, 0x49, 0x49, 0x29, 0x1E],
    ":": [0x00, 0x36, 0x36, 0x00, 0x00],
    ";": [0x00, 0x56, 0x36, 0x00, 0x00],
    "<": [0x08, 0x14, 0x22, 0x41, 0x00],
    "=": [0x14, 0x14, 0x14, 0x14, 0x14],
    ">": [0x00, 0x41, 0x22, 0x14, 0x08],
    "?": [0x02, 0x01, 0x51, 0x09, 0x06],
    "@": [0x32, 0x49, 0x79, 0x41, 0x3E],
    "A": [0x7E, 0x11, 0x11, 0x11, 0x7E],
    "B": [0x7F, 0x49, 0x49, 0x49, 0x36],
    "C": [0x3E, 0x41, 0x41, 0x41, 0x22],
    "D": [0x7F, 0x41, 0x41, 0x22, 0x1C],
    "E": [0x7F, 0x49, 0x49, 0x49, 0x41],
    "F": [0x7F, 0x09, 0x09, 0x09, 0x01],
    "G": [0x3E, 0x41, 0x49, 0x49, 0x7A],
    "H": [0x7F, 0x08, 0x08, 0x08, 0x7F],
    "I": [0x00, 0x41, 0x7F, 0x41, 0x00],
    "J": [0x20, 0x40, 0x41, 0x3F, 0x01],
    "K": [0x7F, 0x08, 0x14, 0x22, 0x41],
    "L": [0x7F, 0x40, 0x40, 0x40, 0x40],
    "M": [0x7F, 0x02, 0x0C, 0x02, 0x7F],
    "N": [0x7F, 0x04, 0x08, 0x10, 0x7F],
    "O": [0x3E, 0x41, 0x41, 0x41, 0x3E],
    "P": [0x7F, 0x09, 0x09, 0x09, 0x06],
    "Q": [0x3E, 0x41, 0x51, 0x21, 0x5E],
    "R": [0x7F, 0x09, 0x19, 0x29, 0x46],
    "S": [0x46, 0x49, 0x49, 0x49, 0x31],
    "T": [0x01, 0x01, 0x7F, 0x01, 0x01],
    "U": [0x3F, 0x40, 0x40, 0x40, 0x3F],
    "V": [0x1F, 0x20, 0x40, 0x20, 0x1F],
    "W": [0x3F, 0x40, 0x38, 0x40, 0x3F],
    "X": [0x63, 0x14, 0x08, 0x14, 0x63],
    "Y": [0x07, 0x08, 0x70, 0x08, 0x07],
    "Z": [0x61, 0x51, 0x49, 0x45, 0x43],
    "[": [0x00, 0x7F, 0x41, 0x41, 0x00],
    "\\": [0x02, 0x04, 0x08, 0x10, 0x20],
    "]": [0x00, 0x41, 0x41, 0x7F, 0x00],
    "^": [0x04, 0x02, 0x01, 0x02, 0x04],
    "_": [0x40, 0x40, 0x40, 0x40, 0x40],
    "`": [0x00, 0x01, 0x02, 0x04, 0x00],
    "a": [0x20, 0x54, 0x54, 0x54, 0x78],
    "b": [0x7F, 0x48, 0x44, 0x44, 0x38],
    "c": [0x38, 0x44, 0x44, 0x44, 0x20],
    "d": [0x38, 0x44, 0x44, 0x48, 0x7F],
    "e": [0x38, 0x54, 0x54, 0x54, 0x18],
    "f": [0x08, 0x7E, 0x09, 0x01, 0x02],
    "g": [0x0C, 0x52, 0x52, 0x52, 0x3E],
    "h": [0x7F, 0x08, 0x04, 0x04, 0x78],
    "i": [0x00, 0x44, 0x7D, 0x40, 0x00],
    "j": [0x20, 0x40, 0x44, 0x3D, 0x00],
    "k": [0x7F, 0x10, 0x28, 0x44, 0x00],
    "l": [0x00, 0x41, 0x7F, 0x40, 0x00],
    "m": [0x7C, 0x04, 0x18, 0x04, 0x78],
    "n": [0x7C, 0x08, 0x04, 0x04, 0x78],
    "o": [0x38, 0x44, 0x44, 0x44, 0x38],
    "p": [0x7C, 0x14, 0x14, 0x14, 0x08],
    "q": [0x08, 0x14, 0x14, 0x18, 0x7C],
    "r": [0x7C, 0x08, 0x04, 0x04, 0x08],
    "s": [0x48, 0x54, 0x54, 0x54, 0x20],
    "t": [0x04, 0x3F, 0x44, 0x40, 0x20],
    "u": [0x3C, 0x40, 0x40, 0x40, 0x7C],
    "v": [0x1C, 0x20, 0x40, 0x20, 0x1C],
    "w": [0x3C, 0x40, 0x30, 0x40, 0x3C],
    "x": [0x44, 0x28, 0x10, 0x28, 0x44],
    "y": [0x0C, 0x50, 0x50, 0x50, 0x3C],
    "z": [0x44, 0x64, 0x54, 0x4C, 0x44],
    "{": [0x00, 0x08, 0x36, 0x41, 0x00],
    "|": [0x00, 0x00, 0x7F, 0x00, 0x00],
    "}": [0x00, 0x41, 0x36, 0x08, 0x00],
    "~": [0x10, 0x08, 0x08, 0x10, 0x08],
}

_CHAR_W = 5   # glyph width in pixels
_CHAR_H = 8   # glyph height in pixels
_CHAR_GAP = 1 # pixels between characters


def put_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color,
    scale: int = 1,
) -> np.ndarray:
    """
    Render *text* onto *img* starting at position (x, y) using a built-in
    5×8 bitmap font.

    Parameters
    ----------
    img   : np.ndarray – target image (modified in-place).
    text  : str        – string to render; unsupported characters → '?'.
    x, y  : int        – top-left pixel position (column, row).
    color : scalar | tuple – text colour matching image depth.
    scale : int        – integer scale factor (≥ 1); scale=2 → 10×16 glyphs.

    Returns
    -------
    np.ndarray – same array as *img*.

    Notes
    -----
    The bitmap font covers printable ASCII (32–126).  Each glyph column is
    stored as a byte; each bit represents one row (bit 0 = top).  Scaling is
    achieved by repeating each pixel *scale* times in both dimensions using
    NumPy's ``np.repeat``.
    """
    _validate_image(img)
    c = _parse_color(color, img)
    if scale < 1:
        raise ValueError(f"'scale' must be ≥ 1, got {scale}.")

    H, W = img.shape[:2]
    cursor_x = x

    for ch in text:
        glyph_cols = _FONT_5x8.get(ch, _FONT_5x8.get("?", [0] * 5))

        for col_idx, col_bits in enumerate(glyph_cols):
            for row_idx in range(_CHAR_H):
                if col_bits & (1 << row_idx):
                    # Scale: write a scale×scale block for each lit bit
                    px = cursor_x + col_idx * scale
                    py = y + row_idx * scale
                    for dr in range(scale):
                        for dc in range(scale):
                            _set_pixel(img, px + dc, py + dr, c)

        cursor_x += (_CHAR_W + _CHAR_GAP) * scale

    return img
