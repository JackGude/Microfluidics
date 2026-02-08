# MNIST Feature Bank

This repo is a feature-extraction project with MNIST where each image is turned into a set of **1D channels**. The main goal is to identify a **compact 3-filter representation** that still yields strong **classification accuracy**.

This README is a reference for:

- what the filters are,
- what each one is measuring (scientifically),
- and which scripts exist.

## Data + preprocessing

- **Dataset**: MNIST handwritten digits.
- **Cropping**: remove a 2-pixel border (`28x28` → `24x24`).
- **Normalization**: pixel intensities scaled to a consistent range (so “mass”/energy and comparisons across images are meaningful).

The output of preprocessing is a `24x24` grayscale image `I(x, y)`.

### Coordinate conventions

- Image coordinates use `x = 0..23` left-to-right and `y = 0..23` top-to-bottom.
- The geometric center is approximately `(x, y) = (11.5, 11.5)`.

## Feature Rationale

### -- Total Filters: 14 --

Most filters below produce a **1D waveform** (a sequence of numbers) because downstream constraints require 1D channels. A few are naturally 2D (e.g., FFT heatmaps), but we still store their low-frequency coefficients as a vector.

You can think of these as different “measurement operators” applied to `I(x, y)`:

- some measure **where ink is** (projections / radial summaries / moments),
- some measure **periodicity / global shape** (Fourier),
- some measure **edges / curvature** (Sobel, Laplacian).

### A) Axis-aligned density projections

- **`x_density`** (length 24)
  - Definition: `x_density[x] = Σ_y I(x, y)`
  - Intuition: vertical “column mass” profile. Captures left/right stroke placement (e.g., distinguishing `1` vs `0`).

- **`y_density`** (length 24)
  - Definition: `y_density[y] = Σ_x I(x, y)`
  - Intuition: horizontal “row mass” profile. Captures top/bottom structure (e.g., `7` has top bar; `9` has upper loop).

### B) Diagonal projections

- **`diag_tlbr`** (length 47)
  - Definition: sums over TL→BR diagonals (constant `x - y`).
  - Intuition: sensitive to “/” vs “\” slanting strokes and diagonal structure.

- **`diag_trbl`** (length 47)
  - Definition: sums over TR→BL diagonals (constant `x + y`).
  - Intuition: complementary diagonal sensitivity; helps distinguish similar digits with different slants.

### C) Fourier features (global shape / periodicity)

- **`fft1d_x`** (length 10)
  - Input: `x_density`.
  - Definition: low-frequency Fourier components for harmonics `k = 1..5`, stored as `[cos(k), ..., sin(k), ...]`.
  - Intuition: compresses the overall “shape” of the x-projection into coarse modes (global left/right balance, multi-lobe structure).

- **`fft1d_y`** (length 10)
  - Same idea, but applied to `y_density`.

- **`fft2_real`** / **`fft2_imag`** (length 16 each)
  - Input: the 2D image `I(x, y)`.
  - Definition: low-frequency 2D Fourier coefficients on a `4x4` grid of `(kx, ky) = 0..3 × 0..3`, split into real and imaginary parts.
  - Intuition: captures global layout (coarse spatial frequencies). This is a “global shape prior” that can separate classes with different large-scale structure.
  - Note: for visualization we often look at **magnitude**, but the model uses the stored real/imag coefficients.

### D) Edge and curvature summaries (after filtering, still exported as 1D)

- **`sobel_mag_xproj`** / **`sobel_mag_yproj`** (length 24 each)
  - Definition: compute Sobel gradients, take gradient magnitude `|∇I|`, then sum along one axis to make a 1D profile.
  - Intuition: “where are the edges?” Summarizes stroke boundaries and their spatial distribution.

- **`lap_abs_xproj`** / **`lap_abs_yproj`** (length 24 each)
  - Definition: apply Laplacian (second-derivative) operator, take absolute response, then project along an axis.
  - Intuition: emphasizes curvature / rapid intensity changes (thin strokes, corners, tight loops).

### E) Radial profile (center-out summaries)

- **`radial_sum`** (12 bins by default)
  - Definition: sum intensity within concentric radial bins around the image center.
  - Intuition: captures “center vs periphery” ink distribution (e.g., loops vs straight strokes).

- **`radial_count`**
  - Definition: number of pixels in each radial bin.
  - Note: constant across images; useful only if you want to compute mean intensity per bin as `radial_sum / radial_count`.

### F) Center-of-mass summary statistics (very low-dimensional)

- **`com_stats`** (length 5)
  - `mass`: total ink `Σ_{x,y} I(x, y)`
  - `(x_mean, y_mean)`: intensity-weighted center of mass
  - `(x_std, y_std)`: intensity-weighted spread along x and y
  - Intuition: crude but interpretable: “where is the digit located” and “how wide/tall is it”.

## How “best 3 filters” is determined

We treat filter selection as a supervised feature-selection problem:

- Choose a subset of filter blocks (e.g., 3 filters).
- Concatenate their outputs into a feature vector.
- Train a simple linear classifier (logistic regression) and measure classification accuracy.

This gives a pragmatic answer to “which measurements are most informative?” under the constraint that we only keep a few channels.

-- 2/7 --

We evaluate classification accuracy for individual filters and drop the three filters that score below 50% accuracy. Then we take the remaining 11 filters and evaluate all possible combinations (165 total) of 3 filters to find the best 3-filter set.

## Scripts (quick reference)

- **`main.py`**
  - MNIST download + preprocessing; writes a small grid image `mnist_examples.png`.

- **`export_to_excel.py`**
  - Extracts the full filter bank and writes an Excel workbook with one sheet per feature.

- **`select_filters.py`**
  - Evaluates supervised classification accuracy for individual filters and for filter combinations (including exhaustive search for the best 3-filter set).

- **`visualize_filters.py`**
  - Generates example visualizations and exports 1D waveforms for presentation (`team_viz/`).

## Excel export output (structure, not usage)

`export_to_excel.py` writes an `.xlsx` file where each **feature is stored on its own sheet**.

All sheets use the same column convention:

- **Column A**: `row_id` (0-based index)
- **Column B**: `label` (MNIST digit 0-9)
- **Columns C+**: feature values
