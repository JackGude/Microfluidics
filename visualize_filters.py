#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.mnist_loader import MNISTProcessor
from src.features.density_features import DensityFeatureExtractor
from src.features.diagonal_features import DiagonalProjectionExtractor
from src.features.edge_features import LaplacianProjectionExtractor, SobelProjectionExtractor
from src.features.fourier_features import Fourier1DDensityExtractor, Fourier2DLowFreqExtractor
from src.features.moment_features import CenterOfMassExtractor
from src.features.radial_features import RadialProfileExtractor


def _parse_digits(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    for d in out:
        if d < 0 or d > 9:
            raise ValueError("digits must be between 0 and 9")
    return out


def _conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if kernel.shape != (3, 3):
        raise ValueError("kernel must be 3x3")

    h, w = image.shape
    padded = np.pad(image, ((1, 1), (1, 1)), mode="constant")

    out = np.zeros((h, w), dtype=float)
    for dy in range(3):
        for dx in range(3):
            k = float(kernel[dy, dx])
            if k == 0.0:
                continue
            out += k * padded[dy : dy + h, dx : dx + w]

    return out


def _edge_maps(image: np.ndarray) -> Dict[str, np.ndarray]:
    gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)

    gx = _conv2d(image, gx_kernel)
    gy = _conv2d(image, gy_kernel)
    mag = np.sqrt(gx * gx + gy * gy)

    lap = _conv2d(image, lap_kernel)
    lap_abs = np.abs(lap)

    return {
        "sobel_gx": gx,
        "sobel_gy": gy,
        "sobel_mag": mag,
        "lap": lap,
        "lap_abs": lap_abs,
    }


def _save_waveform_csv(path: str, x: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mat = np.stack([x.astype(float), y.astype(float)], axis=1)
    np.savetxt(path, mat, delimiter=",", header="idx,value", comments="")


def _save_vector_csv(path: str, v: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx = np.arange(v.shape[0], dtype=int)
    mat = np.stack([idx.astype(float), v.astype(float)], axis=1)
    np.savetxt(path, mat, delimiter=",", header="idx,value", comments="")


def _plot_waveform(ax, v: np.ndarray, title: str) -> None:
    ax.plot(np.arange(v.shape[0]), v, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)


def _plot_heatmap(ax, mat: np.ndarray, title: str) -> None:
    im = ax.imshow(mat, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    import matplotlib.pyplot as plt  # type: ignore

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _select_examples(images: np.ndarray, labels: np.ndarray, digits: Sequence[int], n_per_digit: int) -> List[int]:
    idxs: List[int] = []
    for d in digits:
        found = np.where(labels == int(d))[0]
        if found.size == 0:
            continue
        take = found[: int(n_per_digit)].tolist()
        idxs.extend(take)
    return idxs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        logger.error("matplotlib is not installed; visualize_filters.py requires it")
        logger.error("Activate your venv and install dependencies (or use your OS package manager)")
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--digits", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-per-digit", type=int, default=1)
    parser.add_argument("--output-dir", default="filter_viz")
    args = parser.parse_args()

    logger.info("Starting visualization")
    logger.info("split=%s digits=%s n_per_digit=%d output_dir=%s", args.split, args.digits, int(args.n_per_digit), args.output_dir)

    digits = _parse_digits(str(args.digits))
    if int(args.n_per_digit) <= 0:
        raise ValueError("--n-per-digit must be positive")

    processor = MNISTProcessor(crop_size=2, normalize=True)
    images_t, labels_t = processor.load_mnist(train=(args.split == "train"), download=True)
    images = images_t.numpy()
    labels = labels_t.numpy().astype(int)

    sel = _select_examples(images, labels, digits, int(args.n_per_digit))
    if not sel:
        raise ValueError("No examples selected")

    sel_images = images[sel]
    sel_labels = labels[sel]

    extractors = [
        DensityFeatureExtractor(),
        DiagonalProjectionExtractor(),
        Fourier1DDensityExtractor(),
        Fourier2DLowFreqExtractor(),
        SobelProjectionExtractor(),
        LaplacianProjectionExtractor(use_abs=True),
        CenterOfMassExtractor(),
        RadialProfileExtractor(n_bins=12),
    ]

    feature_blocks: Dict[str, np.ndarray] = {}
    for ex in extractors:
        out = ex.extract_batch(sel_images)
        for k, v in out.items():
            if not isinstance(v, np.ndarray):
                continue
            feature_blocks[k] = v

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(sel_images.shape[0]):
        img = sel_images[i]
        lab = int(sel_labels[i])

        sample_dir = os.path.join(args.output_dir, f"digit_{lab}_row_{int(sel[i])}")
        os.makedirs(sample_dir, exist_ok=True)
        data_dir = os.path.join(sample_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        edges = _edge_maps(img)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.reshape(2, 3)

        _plot_heatmap(axes[0, 0], img, "image (24x24)")
        _plot_heatmap(axes[0, 1], edges["sobel_mag"], "sobel magnitude")
        _plot_heatmap(axes[0, 2], edges["lap_abs"], "laplacian |response|")

        fft2_real = feature_blocks["fft2_real"][i].reshape(4, 4)
        fft2_imag = feature_blocks["fft2_imag"][i].reshape(4, 4)
        fft2_mag = np.sqrt(fft2_real * fft2_real + fft2_imag * fft2_imag)
        _plot_heatmap(axes[1, 0], fft2_real, "fft2 real (4x4 low-freq)")
        _plot_heatmap(axes[1, 1], fft2_imag, "fft2 imag (4x4 low-freq)")
        _plot_heatmap(axes[1, 2], fft2_mag, "fft2 magnitude (4x4 low-freq)")

        fig.suptitle(f"Digit {lab} (row_id={int(sel[i])})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(sample_dir, "overview_2d.png"), dpi=160)
        plt.close(fig)

        waveforms = [
            ("x_density", feature_blocks["x_density"][i]),
            ("y_density", feature_blocks["y_density"][i]),
            ("diag_tlbr", feature_blocks["diag_tlbr"][i]),
            ("diag_trbl", feature_blocks["diag_trbl"][i]),
            ("fft1d_x", feature_blocks["fft1d_x"][i]),
            ("fft1d_y", feature_blocks["fft1d_y"][i]),
            ("fft2_real", feature_blocks["fft2_real"][i]),
            ("fft2_imag", feature_blocks["fft2_imag"][i]),
            ("sobel_mag_xproj", feature_blocks["sobel_mag_xproj"][i]),
            ("sobel_mag_yproj", feature_blocks["sobel_mag_yproj"][i]),
            ("lap_abs_xproj", feature_blocks["lap_abs_xproj"][i]),
            ("lap_abs_yproj", feature_blocks["lap_abs_yproj"][i]),
            ("radial_sum", feature_blocks["radial_sum"][i]),
            ("com_stats", feature_blocks["com_stats"][i]),
        ]

        n_panels = len(waveforms)
        n_cols = 2
        n_rows = int(np.ceil(n_panels / float(n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.6 * n_rows))
        axes = np.atleast_2d(axes)

        for j, (name, v) in enumerate(waveforms):
            ax = axes[j // 2, j % 2]
            _plot_waveform(ax, np.asarray(v).astype(float), name)
            _save_vector_csv(os.path.join(data_dir, f"{name}.csv"), np.asarray(v))

        for k in range(n_panels, n_rows * n_cols):
            ax = axes[k // 2, k % 2]
            ax.axis("off")

        fig.suptitle(f"Digit {lab} waveforms")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(sample_dir, "waveforms_1d.png"), dpi=160)
        plt.close(fig)

        np.save(os.path.join(data_dir, "image.npy"), img.astype(np.float32, copy=False))
        np.save(os.path.join(data_dir, "sobel_mag.npy"), edges["sobel_mag"].astype(np.float32, copy=False))
        np.save(os.path.join(data_dir, "lap_abs.npy"), edges["lap_abs"].astype(np.float32, copy=False))

    logger.info("Saved filter visualizations to: %s", args.output_dir)
    logger.info("Each sample folder contains:")
    logger.info("- overview_2d.png (image + edge maps + FFT heatmaps)")
    logger.info("- waveforms_1d.png (all 1D channels as waveforms)")
    logger.info("- data/ (CSV waveforms + NPY arrays)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
