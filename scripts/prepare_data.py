
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


import argparse

parser = argparse.ArgumentParser(
    description="Prepare mouse V1 spike-count features from Allen NWB data."
)

parser.add_argument(
    "--data-dir",
    type=Path,
    required=True,
    help="Directory containing channels.csv and the NWB session folder.",
)

args = parser.parse_args()

DATA_DIR = args.data_dir
NWB_PATH = DATA_DIR / "session_757216464" / "session_757216464.nwb"
CHANNELS_PATH = DATA_DIR / "channels.csv"


# Output inside the GitHub project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

channels = pd.read_csv(CHANNELS_PATH)

channel_to_region = dict(
    zip(channels["id"], channels["ecephys_structure_acronym"])
)

with h5py.File(NWB_PATH, "r") as f:
    gratings = f["intervals"]["drifting_gratings_presentations"]

    stimulus_df = pd.DataFrame({
        "orientation": gratings["orientation"][:],
        "start_time": gratings["start_time"][:],
        "stop_time": gratings["stop_time"][:],
    })

    stimulus_df = (
        stimulus_df.dropna(subset=["orientation"])
        .reset_index(drop=True)
    )

    stimulus_df["orientation"] = stimulus_df["orientation"].astype(int)

    units = f["units"]
    peak_channel_ids = units["peak_channel_id"][:]
    spike_times = units["spike_times"][:]
    spike_index = units["spike_times_index"][:]

    unit_regions = np.array([
        channel_to_region.get(channel_id, "unknown")
        for channel_id in peak_channel_ids
    ])

    visp_indices = np.where(unit_regions == "VISp")[0]

    print("VISp neurons:", len(visp_indices))
    print("Trials:", len(stimulus_df))

    X = np.zeros(
        (len(stimulus_df), len(visp_indices)),
        dtype=np.float32
    )

    for neuron_idx, unit_index in enumerate(visp_indices):
        start_idx = 0 if unit_index == 0 else spike_index[unit_index - 1]
        end_idx = spike_index[unit_index]

        spikes = spike_times[start_idx:end_idx]

        for trial_idx, trial in stimulus_df.iterrows():
            X[trial_idx, neuron_idx] = np.count_nonzero(
                (spikes >= trial["start_time"])
                & (spikes < trial["stop_time"])
            )

y = stimulus_df["orientation"].to_numpy()

np.savez_compressed(
    OUTPUT_DIR / "spike_counts.npz",
    X=X,
    y=y
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Directions:", np.unique(y))
print("Saved:", OUTPUT_DIR / "spike_counts.npz")
