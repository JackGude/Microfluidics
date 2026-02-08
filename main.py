#!/usr/bin/env python3
"""
Main demonstration script for MNIST loading / preprocessing.
"""

import sys
import os
import logging
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.mnist_loader import MNISTProcessor


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def demonstrate_mnist_loading():
    """Demonstrate MNIST download + preprocessing."""
    logger = logging.getLogger(__name__)
    logger.info("=== MNIST Loading Demo ===")

    processor = MNISTProcessor(crop_size=2, normalize=True)

    logger.info("Loading MNIST dataset...")
    images, labels = processor.load_mnist(train=True, download=True)
    logger.info("Loaded %d images", len(images))

    info = processor.get_sample_info()
    logger.info("Dataset Info:")
    logger.info("  Original size: %s", info["original_size"])
    logger.info("  Processed size: %s", info["processed_size"])
    logger.info("  Normalized: %s", info["normalized"])

    logger.info("Tensor shapes:")
    logger.info("  images: %s", tuple(images.shape))
    logger.info("  labels: %s", tuple(labels.shape))

    return processor, images, labels


def visualize_examples(images: np.ndarray, labels: np.ndarray, *, out_path: str) -> None:
    logger = logging.getLogger(__name__)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping image grid output (%s)", out_path)
        return

    n = min(16, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i], cmap="gray")
            ax.set_title(str(int(labels[i])))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Wrote: %s", out_path)


if __name__ == "__main__":
    try:
        _setup_logging()
        processor, images, labels = demonstrate_mnist_loading()
        visualize_examples(images[:16].numpy(), labels[:16].numpy(), out_path="mnist_examples.png")
    except Exception as e:
        _setup_logging()
        logging.getLogger(__name__).exception("Error: %s", e)
        sys.exit(1)
