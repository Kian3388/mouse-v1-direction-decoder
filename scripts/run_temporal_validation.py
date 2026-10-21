
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


import argparse

parser = argparse.ArgumentParser(
    description="Evaluate mouse V1 decoding on chronologically separated trials."
)

parser.add_argument(
    "--data-dir",
    type=Path,
    required=True,
    help="Directory containing the original NWB session folder.",
)

args = parser.parse_args()

DATA_DIR = args.data_dir


NWB_PATH = (
    DATA_DIR
    / "session_757216464"
    / "session_757216464.nwb"
)

# Load the reconstructed features
data = np.load(PROJECT_ROOT / "data/processed/spike_counts.npz")
X = data["X"]
y = data["y"]

# Verify chronological ordering
with h5py.File(NWB_PATH, "r") as f:
    gratings = f["intervals"]["drifting_gratings_presentations"]
    directions = gratings["orientation"][:]
    start_times = gratings["start_time"][:]

valid = ~np.isnan(directions)
start_times = start_times[valid]
directions = directions[valid].astype(int)

assert len(start_times) == len(y)
assert np.array_equal(directions, y), "Stimulus labels do not match!"
assert np.all(np.diff(start_times) >= 0), "Trials are not chronological!"

print("Chronological ordering verified.")

# First 80% for training, final 20% for testing
split = int(len(y) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000, random_state=42)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTemporal validation results")
print("---------------------------")
print("Training trials:", len(y_train))
print("Testing trials:", len(y_test))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Balanced accuracy:", balanced_accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
