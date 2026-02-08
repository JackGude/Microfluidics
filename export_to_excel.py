#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.mnist_loader import MNISTProcessor
from src.export.excel_exporter import export_features_to_excel
from src.features.density_features import DensityFeatureExtractor
from src.features.diagonal_features import DiagonalProjectionExtractor
from src.features.fourier_features import Fourier1DDensityExtractor, Fourier2DLowFreqExtractor
from src.features.edge_features import SobelProjectionExtractor, LaplacianProjectionExtractor
from src.features.moment_features import CenterOfMassExtractor
from src.features.radial_features import RadialProfileExtractor


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="mnist_features.xlsx")
    args = parser.parse_args()

    logger.info("Starting Excel export")
    logger.info("split=%s limit=%s output=%s", args.split, args.limit, args.output)

    processor = MNISTProcessor(crop_size=2, normalize=True)
    density_extractor = DensityFeatureExtractor()
    diagonal_extractor = DiagonalProjectionExtractor()
    fourier_1d_extractor = Fourier1DDensityExtractor()
    fourier_2d_extractor = Fourier2DLowFreqExtractor()
    sobel_extractor = SobelProjectionExtractor()
    laplacian_extractor = LaplacianProjectionExtractor(use_abs=True)
    com_extractor = CenterOfMassExtractor()
    radial_extractor = RadialProfileExtractor(n_bins=12)

    images_list = []
    labels_list = []

    if args.split in ("train", "all"):
        logger.info("Loading train split")
        images, labels = processor.load_mnist(train=True, download=True)
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())

    if args.split in ("test", "all"):
        logger.info("Loading test split")
        images, labels = processor.load_mnist(train=False, download=True)
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())

    all_images = np.concatenate(images_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    if args.limit and args.limit > 0:
        all_images = all_images[: args.limit]
        all_labels = all_labels[: args.limit]

    logger.info("Extracting features for %d samples", int(all_images.shape[0]))

    density_features = density_extractor.extract_batch(all_images)
    diagonal_features = diagonal_extractor.extract_batch(all_images)
    fourier_1d_features = fourier_1d_extractor.extract_batch(all_images)
    fourier_2d_features = fourier_2d_extractor.extract_batch(all_images)
    sobel_features = sobel_extractor.extract_batch(all_images)
    laplacian_features = laplacian_extractor.extract_batch(all_images)
    com_features = com_extractor.extract_batch(all_images)
    radial_features = radial_extractor.extract_batch(all_images)

    export_features_to_excel(
        labels=all_labels,
        feature_dicts=[
            density_features,
            diagonal_features,
            fourier_1d_features,
            fourier_2d_features,
            sobel_features,
            laplacian_features,
            com_features,
            radial_features,
        ],
        output_path=args.output,
    )

    logger.info("Wrote: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
