"""Run cross-validated analyses from a saved trial-by-neuron matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mouse_v1_decoder import (  # noqa: E402
    direction_to_orientation,
    evaluate_decoder,
    permutation_baseline,
    population_size_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "spike_counts.npz",
        help="NPZ file containing arrays named X and y",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "extended",
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def save_confusion(matrix: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(args.features)
    features = arrays["X"]
    direction_labels = arrays["y"]

    direction_metrics, direction_cm, _ = evaluate_decoder(features, direction_labels)
    orientation_labels = direction_to_orientation(direction_labels)
    orientation_metrics, orientation_cm, _ = evaluate_decoder(features, orientation_labels)

    permutation = permutation_baseline(
        features,
        direction_labels,
        n_permutations=args.permutations,
    )

    default_sizes = [size for size in [5, 10, 25, 50, 100, 200, features.shape[1]] if size <= features.shape[1]]
    curve = pd.DataFrame(
        population_size_curve(
            features,
            direction_labels,
            default_sizes,
            repeats=args.repeats,
        )
    )
    curve.to_csv(args.output_dir / "population_size_results.csv", index=False)

    summary = {
        "direction": direction_metrics.to_dict(),
        "orientation": orientation_metrics.to_dict(),
        "permutation_test": permutation,
        "n_trials": int(features.shape[0]),
        "n_neurons": int(features.shape[1]),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_confusion(
        direction_cm,
        np.unique(direction_labels),
        "Out-of-fold direction decoding",
        args.output_dir / "direction_confusion_matrix.png",
    )
    save_confusion(
        orientation_cm,
        np.unique(orientation_labels),
        "Out-of-fold orientation decoding",
        args.output_dir / "orientation_confusion_matrix.png",
    )

    summary_curve = curve.groupby("population_size")["accuracy"].agg(["mean", "std"])
    plt.errorbar(
        summary_curve.index,
        summary_curve["mean"],
        yerr=summary_curve["std"],
        marker="o",
        capsize=3,
    )
    plt.axhline(1 / np.unique(direction_labels).size, color="black", linestyle="--", label="chance")
    plt.xlabel("Number of sampled VISp units")
    plt.ylabel("Cross-validated accuracy")
    plt.title("Direction decoding by population size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "population_size_curve.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
