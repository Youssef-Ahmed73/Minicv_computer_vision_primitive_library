# minicv

A reusable Python image-processing library that emulates a well-defined subset of OpenCV — built **from scratch** using only **NumPy**, **Pandas** (indirectly via NumPy conventions), **Matplotlib**, and the **Python standard library**.

---

## Project Structure

```
minicv/
├── minicv/               ← importable package
│   ├── __init__.py       ← public API surface
│   ├── utils.py          ← validation, dtype cast, normalize, clip, pad, convolve2d
│   ├── io.py             ← read_image, export_image, rgb_to_gray, gray_to_rgb
│   ├── filters.py        ← mean, gaussian, median, threshold, sobel, bit-plane, histogram, laplacian, unsharp
│   ├── transforms.py     ← resize, rotate, translate
│   ├── features.py       ← color_histogram, pixel_statistics, gradient_histogram, lbp
│   └── drawing.py        ← draw_point, draw_line, draw_rectangle, draw_polygon, put_text
├── tests/
│   └── test_all.py       ← 32 unit tests + visual result generation
└── README.md
```

---

## Installation & Usage

```python
# No install needed — just add the repo root to your Python path:
import sys
sys.path.insert(0, "/path/to/minicv")

import minicv
from minicv import io, utils, filters, transforms, features, drawing
import numpy as np

# Load an image
img = io.read_image("photo.jpg")          # → float64 (H,W,3) in [0,1]

# Convert to grayscale
gray = io.rgb_to_gray(img)                # → float64 (H,W)

# Smooth
smoothed = filters.gaussian_filter(gray, size=7, sigma=1.5)

# Detect edges
grads = filters.sobel_gradients(gray)
edges = grads["magnitude"]

# Resize
small = transforms.resize(img, 128, 128, "bilinear")

# Draw on a canvas
canvas = np.zeros((256, 256, 3), dtype=np.float64)
drawing.draw_rectangle(canvas, 10, 10, 100, 100, (1,0,0), thickness=2)
drawing.put_text(canvas, "Hello!", 15, 110, (1,1,1), scale=2)

# Save result
io.export_image(canvas, "output.png")
```

---

## Math & Algorithms Notes

### Convolution (utils.convolve2d)

True 2-D cross-correlation (same as OpenCV `filter2D`):

```
(f ⋆ k)[i,j] = Σ_m Σ_n  f[i+m, j+n] · k[m, n]
```

Implementation uses NumPy `as_strided` to build a 4-D sliding-window view `(H, W, kH, kW)` without copying data, then contracts with `einsum("ijkl,kl->ij", windows, kernel)`. **Zero Python loops over pixels.**

### Padding Modes

| Mode | Border rule | Use case |
|------|-------------|----------|
| `constant` | Fill with scalar (default 0) | Convolution preserving size |
| `reflect` | `d c b | a b c d | c b a` | Smooth natural images |
| `replicate` | `a a a | a b c d | d d d` | Avoids dark edge artefacts |

### Normalization

| Mode | Formula | Output range |
|------|---------|--------------|
| `minmax` | `(x − min) / (max − min)` | [0, 1] |
| `zscore` | `(x − μ) / σ` | zero mean, unit std |
| `unit` | `x / ‖x‖₂` | unit L2 norm |

### Gaussian Kernel

```
G(x,y) = exp(−(x² + y²) / 2σ²)    then normalise to sum = 1
```

Grid is centred at (0,0) using `np.meshgrid(arange(-half, half+1), ...)`.

### Otsu Thresholding

Maximises **between-class variance** over all thresholds t ∈ [0, 255]:

```
σ²_B(t) = ω₀(t) · ω₁(t) · [μ₀(t) − μ₁(t)]²
```

where ω₀, ω₁ = class probabilities, μ₀, μ₁ = class means. Computed fully vectorised over 256 bins.

### Adaptive Thresholding

```
thresh(i,j) = mean(neighbourhood(i,j)) − C
output(i,j) = 1  if img(i,j) > thresh(i,j)  else 0
```

Local mean computed via convolution (mean or Gaussian kernel). Runs in O(H·W) time.

### Sobel Gradients

```
Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]    ← horizontal gradient
Ky = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]    ← vertical gradient

magnitude = √(Gx² + Gy²)
angle     = arctan2(Gy, Gx)
```

### Bit-Plane Slicing

```
plane_image[i,j] = (uint8_pixel[i,j] >> n) & 1
```

where n=7 is the MSB (most structural information), n=0 is the LSB (near-random for natural images).

### Histogram Equalisation

```
1. Compute histogram p(k) for k=0..255
2. CDF: cdf(k) = Σ_{j≤k} p(j)
3. out[i,j] = cdf(img[i,j])   (normalised to [0,1])
```

### Geometric Transforms — Inverse Mapping

All transforms use **inverse mapping** to avoid holes:
for each destination pixel, compute the source coordinate, then interpolate.

**Resize:**
```
src_y = (dst_r + 0.5) · (H_src / H_dst) − 0.5
src_x = (dst_c + 0.5) · (W_src / W_dst) − 0.5
```
The ±0.5 aligns pixel centres between grids (same as OpenCV `INTER_LINEAR`).

**Bilinear interpolation:**
```
f(y,x) ≈ (1−dy)(1−dx)·f(y0,x0) + (1−dy)·dx·f(y0,x1)
        +    dy·(1−dx)·f(y1,x0) +    dy·dx·f(y1,x1)
```

**Rotation about centre:**
```
[src_x]   [ cosθ  sinθ] [dst_x − cx]   [cx]
[src_y] = [−sinθ  cosθ] [dst_y − cy] + [cy]
```

### Feature Descriptors

**Color Histogram:** Per-channel value histogram, concatenated → global appearance vector.

**Pixel Statistics:** [mean, std, min, max, skewness] per channel. Skewness = E[(x−μ)³]/σ³.

**Gradient Orientation Histogram (HOG-like):**
Magnitude-weighted histogram of Sobel gradient angles. Encodes dominant edge directions.

**Local Binary Pattern (LBP):**
For each pixel, compare N neighbours on a circle of radius R to the centre pixel:
```
LBP(i,j) = Σ_{k=0}^{N-1}  s(nbr_k − centre) · 2^k
```
where s(x)=1 if x≥0. The histogram of all LBP codes is the texture descriptor. Invariant to monotonic illumination changes.

### Drawing Primitives

**Bresenham Line:** Integer-only arithmetic with accumulated error term:
```
error += 2·dy
if error > dx: y += step_y;  error -= 2·dx
```

**Scanline Polygon Fill:** For each scanline y, find all edge intersections, sort, fill between pairs.

**Text:** Built-in 5×8 bitmap font (packed as 5 column bytes per glyph). Scaled by integer repeating.

---

## API Reference

See docstrings in each module for full parameter documentation. Quick reference:

### `minicv.utils`
| Function | Signature | Returns |
|----------|-----------|---------|
| `normalize` | `(img, mode='minmax')` | float64 ndarray |
| `clip_pixels` | `(img, lo=0, hi=1)` | ndarray same dtype |
| `pad` | `(img, pad_h, pad_w, mode='constant')` | float64 ndarray |
| `convolve2d` | `(img, kernel, pad_mode='constant')` | float64 (H,W) |
| `spatial_filter` | `(img, kernel, pad_mode='constant')` | float64 same shape |
| `to_float64` | `(img)` | float64 [0,1] |
| `to_uint8` | `(img)` | uint8 [0,255] |

### `minicv.io`
| Function | Signature | Returns |
|----------|-----------|---------|
| `read_image` | `(path)` | float64 (H,W) or (H,W,3) |
| `export_image` | `(img, path, quality=95)` | None |
| `rgb_to_gray` | `(img)` | float64 (H,W) |
| `gray_to_rgb` | `(img)` | float64 (H,W,3) |

### `minicv.filters`
| Function | Signature |
|----------|-----------|
| `mean_filter` | `(img, kernel_size=3, pad_mode='constant')` |
| `gaussian_kernel` | `(size, sigma) → (size,size) kernel` |
| `gaussian_filter` | `(img, size=5, sigma=1.0, pad_mode='reflect')` |
| `median_filter` | `(img, kernel_size=3, pad_mode='reflect')` |
| `threshold_global` | `(img, thresh, max_val=1.0)` |
| `threshold_otsu` | `(img) → (binary, threshold)` |
| `threshold_adaptive` | `(img, block_size=11, C=0.02, method='mean')` |
| `sobel_gradients` | `(img) → dict(Gx, Gy, magnitude, angle)` |
| `bit_plane_slice` | `(img, plane)` |
| `histogram` | `(img, bins=256, normalized=False) → (counts, edges)` |
| `equalize_histogram` | `(img)` |
| `laplacian_filter` | `(img)` |
| `unsharp_mask` | `(img, sigma=2.0, strength=1.5)` |

### `minicv.transforms`
| Function | Signature |
|----------|-----------|
| `resize` | `(img, new_h, new_w, interpolation='bilinear')` |
| `rotate` | `(img, angle_deg, interpolation='bilinear', fill=0.0)` |
| `translate` | `(img, tx, ty, interpolation='bilinear', fill=0.0)` |

### `minicv.features`
| Function | Returns shape |
|----------|---------------|
| `color_histogram(img, bins=32)` | `(bins,)` gray or `(3·bins,)` RGB |
| `pixel_statistics(img)` | `(5,)` gray or `(15,)` RGB |
| `gradient_histogram(img, bins=9)` | `(bins,)` |
| `lbp(img, radius=1, n_points=8, bins=256)` | `(bins,)` |

### `minicv.drawing`
| Function | Signature |
|----------|-----------|
| `draw_point` | `(img, x, y, color, radius=1)` |
| `draw_line` | `(img, x0, y0, x1, y1, color, thickness=1)` |
| `draw_rectangle` | `(img, x0, y0, x1, y1, color, thickness=1, filled=False)` |
| `draw_polygon` | `(img, points, color, thickness=1, filled=False)` |
| `put_text` | `(img, text, x, y, color, scale=1)` |

---

## Running Tests

```bash
# From the project root:
python tests/test_all.py
```

Expected output: **32 passed, 0 failed**, plus a result image saved to `/tmp/minicv_results.png`.
