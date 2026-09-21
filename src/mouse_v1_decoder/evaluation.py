"""Leakage-safe evaluation tools for neural population decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DecodingMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_xy(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError("features must be a 2D trial-by-neuron matrix")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must contain the same number of trials")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")
    if np.unique(labels).size < 2:
        raise ValueError("at least two label classes are required")

    return features, labels


def make_decoder(random_state: int = 42):
    """Return a scaler-plus-logistic-regression pipeline.

    Keeping scaling inside the pipeline ensures that each cross-validation fold
    learns its scaling parameters from the training portion only.
    """

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3_000, random_state=random_state),
    )


def direction_to_orientation(direction_labels: np.ndarray) -> np.ndarray:
    """Collapse opposite drifting directions onto four orientation axes."""

    labels = np.asarray(direction_labels)
    if not np.isfinite(labels.astype(float)).all():
        raise ValueError("direction labels must be finite numeric values")
    return labels.astype(int) % 180


def evaluate_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[DecodingMetrics, np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions and summary metrics.

    Each trial is predicted by a model that was not trained on that trial.
    """

    features, labels = _validate_xy(features, labels)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions = cross_val_predict(make_decoder(random_state), features, labels, cv=cv)
    metrics = DecodingMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro")),
    )
    classes = np.unique(labels)
    matrix = confusion_matrix(labels, predictions, labels=classes)
    return metrics, matrix, predictions


def permutation_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 200,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare observed accuracy with accuracies after label shuffling."""

    features, labels = _validate_xy(features, labels)
    observed, _, _ = evaluate_decoder(
        features, labels, n_splits=n_splits, random_state=random_state
    )
    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=float)

    for index in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_metrics, _, _ = evaluate_decoder(
            features,
            shuffled,
            n_splits=n_splits,
            random_state=random_state + index + 1,
        )
        null_scores[index] = null_metrics.accuracy

    p_value = (1 + np.sum(null_scores >= observed.accuracy)) / (n_permutations + 1)
    return {
        "observed_accuracy": observed.accuracy,
        "null_mean_accuracy": float(null_scores.mean()),
        "null_std_accuracy": float(null_scores.std(ddof=1)),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
    }


def population_size_curve(
    features: np.ndarray,
    labels: np.ndarray,
    population_sizes: list[int],
    *,
    repeats: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict[str, float | int]]:
    """Estimate decoder performance while varying sampled neuron count."""

    features, labels = _validate_xy(features, labels)
    n_neurons = features.shape[1]
    invalid = [size for size in population_sizes if size < 1 or size > n_neurons]
    if invalid:
        raise ValueError(f"population sizes must be between 1 and {n_neurons}: {invalid}")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []

    for size in sorted(set(population_sizes)):
        for repeat in range(repeats):
            selected = rng.choice(n_neurons, size=size, replace=False)
            metrics, _, _ = evaluate_decoder(
                features[:, selected],
                labels,
                n_splits=n_splits,
                random_state=random_state + repeat,
            )
            rows.append(
                {
                    "population_size": int(size),
                    "repeat": int(repeat),
                    **metrics.to_dict(),
                }
            )

    return rows
