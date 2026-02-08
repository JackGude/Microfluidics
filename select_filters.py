#!/usr/bin/env python3

import argparse
import itertools
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.mnist_loader import MNISTProcessor
from src.features.density_features import DensityFeatureExtractor
from src.features.diagonal_features import DiagonalProjectionExtractor
from src.features.edge_features import LaplacianProjectionExtractor, SobelProjectionExtractor
from src.features.fourier_features import Fourier1DDensityExtractor, Fourier2DLowFreqExtractor
from src.features.moment_features import CenterOfMassExtractor
from src.features.radial_features import RadialProfileExtractor


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("Expected 1D or 2D array")
    return x


def _fit_eval_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    max_iter: int,
    solver: str,
    tol: float,
    n_jobs: int,
) -> float:
    # Keep output readable: some solvers may warn about convergence on small max_iter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs if solver in ("saga", "liblinear") else None,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return float(accuracy_score(y_test, pred))


def _maybe_top_candidates(keys: Sequence[str], scores: Sequence[Tuple[str, float, int]], top_n: int) -> List[str]:
    if top_n <= 0 or top_n >= len(keys):
        return list(keys)
    ranked = [name for (name, _, _) in scores]
    return ranked[:top_n]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--report-top", type=int, default=20)
    parser.add_argument(
        "--exhaustive-top",
        type=int,
        default=0,
        help="If >0, only consider the top-N single-sheet filters as candidates for exhaustive search (dramatically faster).",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "saga", "liblinear"],
        default="lbfgs",
        help="LogisticRegression solver. 'saga' can use multiple CPU cores via --n-jobs.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Optimization tolerance (larger can be faster but slightly less accurate).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Used for some solvers (e.g. saga). -1 uses all cores.",
    )
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    logger.info("Starting filter selection")
    logger.info(
        "mode=%s top_k=%d limit_train=%d limit_test=%d solver=%s C=%s max_iter=%d tol=%s n_jobs=%d",
        "exhaustive" if bool(args.exhaustive) else "greedy",
        int(args.top_k),
        int(args.limit_train),
        int(args.limit_test),
        str(args.solver),
        str(args.C),
        int(args.max_iter),
        str(args.tol),
        int(args.n_jobs),
    )

    processor = MNISTProcessor(crop_size=2, normalize=True)

    train_images, train_labels = processor.load_mnist(train=True, download=True)
    test_images, test_labels = processor.load_mnist(train=False, download=True)

    Xtr_img = train_images.numpy()
    ytr = train_labels.numpy().astype(int)
    Xte_img = test_images.numpy()
    yte = test_labels.numpy().astype(int)

    if args.limit_train and args.limit_train > 0:
        Xtr_img = Xtr_img[: args.limit_train]
        ytr = ytr[: args.limit_train]

    if args.limit_test and args.limit_test > 0:
        Xte_img = Xte_img[: args.limit_test]
        yte = yte[: args.limit_test]

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

    exclude = set(args.exclude)
    exclude.add("radial_count")

    train_features: Dict[str, np.ndarray] = {}
    test_features: Dict[str, np.ndarray] = {}

    extractor_iter = extractors
    if tqdm is not None:
        extractor_iter = tqdm(extractors, desc="Extracting feature blocks", unit="extractor")

    for ex in extractor_iter:
        tr = ex.extract_batch(Xtr_img)
        te = ex.extract_batch(Xte_img)
        for k, v in tr.items():
            if k in exclude:
                continue
            if not isinstance(v, np.ndarray):
                continue
            train_features[k] = _ensure_2d(v.astype(np.float32, copy=False))
        for k, v in te.items():
            if k in exclude:
                continue
            if not isinstance(v, np.ndarray):
                continue
            test_features[k] = _ensure_2d(v.astype(np.float32, copy=False))

    keys = sorted(set(train_features.keys()) & set(test_features.keys()))
    if not keys:
        raise ValueError("No per-sheet feature blocks found")

    scaled_train: Dict[str, np.ndarray] = {}
    scaled_test: Dict[str, np.ndarray] = {}

    for k in keys:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(train_features[k])
        Xte = scaler.transform(test_features[k])
        scaled_train[k] = Xtr
        scaled_test[k] = Xte

    # Single-sheet ranking
    single_scores: List[Tuple[str, float, int]] = []
    key_iter = keys
    if tqdm is not None:
        key_iter = tqdm(keys, desc="Single-sheet eval", unit="sheet")

    for k in key_iter:
        acc = _fit_eval_accuracy(
            scaled_train[k],
            ytr,
            scaled_test[k],
            yte,
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            tol=float(args.tol),
            n_jobs=int(args.n_jobs),
        )
        single_scores.append((k, acc, int(scaled_train[k].shape[1])))

    single_scores.sort(key=lambda t: t[1], reverse=True)

    logger.info("=== Single-sheet accuracy (train->test) ===")
    for name, acc, d in single_scores:
        logger.info("%s  acc=%.4f  dims=%d", f"{name:20s}", float(acc), int(d))

    k = int(args.top_k)
    if k <= 0:
        raise ValueError("--top-k must be >= 1")
    if k > len(keys):
        raise ValueError(f"--top-k={k} exceeds number of available filters ({len(keys)})")

    if args.exhaustive:
        cand_keys = _maybe_top_candidates(keys, single_scores, int(args.exhaustive_top))
        combos = list(itertools.combinations(cand_keys, k))
        logger.info("Exhaustive candidates=%d combos=%d", len(cand_keys), len(combos))
        combo_iter = combos
        if tqdm is not None:
            combo_iter = tqdm(combos, desc=f"Exhaustive {len(combos)} combos", unit="combo")

        scored_combos: List[Tuple[Tuple[str, ...], float, int]] = []
        best_seen: Optional[Tuple[Tuple[str, ...], float, int]] = None
        try:
            for combo in combo_iter:
                Xtr = np.concatenate([scaled_train[x] for x in combo], axis=1)
                Xte = np.concatenate([scaled_test[x] for x in combo], axis=1)
                acc = _fit_eval_accuracy(
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    C=args.C,
                    max_iter=args.max_iter,
                    solver=args.solver,
                    tol=float(args.tol),
                    n_jobs=int(args.n_jobs),
                )
                row = (combo, float(acc), int(Xtr.shape[1]))
                scored_combos.append(row)
                if best_seen is None or row[1] > best_seen[1]:
                    best_seen = row
                    if tqdm is not None and hasattr(combo_iter, "set_postfix"):
                        combo_iter.set_postfix(best_acc=f"{best_seen[1]:.4f}")
        except KeyboardInterrupt:
            logger.warning("Interrupted. Reporting best result seen so far.")
            if best_seen is not None:
                combo, acc, dims = best_seen
                combo_str = ", ".join(combo)
                logger.warning("best_so_far: [%s]  acc=%.4f  total_dims=%d", combo_str, float(acc), int(dims))
            if not scored_combos:
                return 130

        scored_combos.sort(key=lambda t: t[1], reverse=True)

        top_n = max(1, min(int(args.report_top), len(scored_combos)))
        logger.info("=== Top %d combos (train->test) ===", int(top_n))
        for combo, acc, dims in scored_combos[:top_n]:
            combo_str = ", ".join(combo)
            logger.info("[%s]  acc=%.4f  total_dims=%d", combo_str, float(acc), int(dims))

        best_combo, best_acc, best_dims = scored_combos[0]
        logger.info("Best combo:")
        logger.info("acc=%.4f  total_dims=%d", float(best_acc), int(best_dims))
        for name in best_combo:
            logger.info("- %s", name)
    else:
        selected: List[str] = []
        remaining = set(keys)

        logger.info("=== Greedy selection ===")
        for step in range(k):
            best_name = None
            best_acc = -1.0
            best_dims = 0

            candidates = sorted(remaining)
            cand_iter = candidates
            if tqdm is not None:
                cand_iter = tqdm(candidates, desc=f"Greedy step {step+1}", unit="cand", leave=False)

            for cand in cand_iter:
                combo = selected + [cand]
                Xtr = np.concatenate([scaled_train[x] for x in combo], axis=1)
                Xte = np.concatenate([scaled_test[x] for x in combo], axis=1)
                acc = _fit_eval_accuracy(
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    C=args.C,
                    max_iter=args.max_iter,
                    solver=args.solver,
                    tol=float(args.tol),
                    n_jobs=int(args.n_jobs),
                )
                if acc > best_acc:
                    best_acc = acc
                    best_name = cand
                    best_dims = int(Xtr.shape[1])

            if best_name is None:
                break

            selected.append(best_name)
            remaining.remove(best_name)
            logger.info(
                "step %d: add %s -> acc=%.4f, total_dims=%d",
                int(step + 1),
                str(best_name),
                float(best_acc),
                int(best_dims),
            )

        logger.info("Selected filters:")
        for s in selected:
            logger.info("- %s", s)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger(__name__).exception("Fatal error: %s", e)
        raise
