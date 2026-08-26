# Mouse V1 Visual Direction Decoder

A self-directed computational neuroscience project that tests whether drifting-grating motion direction can be decoded from population activity in mouse primary visual cortex (VISp/V1).

The analysis uses real Allen Institute Visual Coding Neuropixels data and a multiclass logistic-regression classifier.

## Current Results

* Allen Institute session: `757216464`
* Usable drifting-grating trials: `600`
* VISp units: `261`
* Direction classes: `8`
* Chance accuracy: `12.5%`
* Baseline test accuracy: `91.7%`

The largest classification errors occurred between 45° and 225°. These are opposite motion directions that share the same grating orientation, motivating a later comparison between eight-direction decoding and four-orientation decoding.

The reported accuracy currently comes from one stratified 80/20 train-test split and should be interpreted as a baseline result rather than a final estimate of generalization performance.

## Scientific Question

Can drifting-grating motion direction be decoded from population spiking activity in mouse primary visual cortex?

## Dataset

This project uses the Allen Institute Brain Observatory Visual Coding Neuropixels dataset.

* Session ID: `757216464`
* Session type: `brain_observatory_1.1`
* Stimulus: drifting gratings
* Brain region: `VISp`

The NWB file is approximately 3 GB and is not included in this repository.

## Method

1. Read drifting-grating presentation times and labels directly from the NWB file with `h5py`.
2. Remove presentations with undefined direction labels.
3. Map units to brain regions using Allen Institute channel metadata.
4. Select units whose peak channels are located in `VISp`.
5. Count each unit's spikes during every stimulus presentation.
6. Use the resulting trial-by-neuron spike-count matrix as the model input.
7. Train a pipeline containing `StandardScaler` and multiclass `LogisticRegression`.
8. Evaluate predictions using accuracy and a confusion matrix.

The final feature matrix contains 600 trials and 261 neural features.

## Repository Structure

```text
mouse-v1-direction-decoder/
├── notebooks/
│   └── explorations.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The `data/` directory and local virtual environments are excluded from GitHub.

## Local Data Structure

To run the notebook locally, the files should be arranged as follows:

```text
data/
├── channels.csv
└── session_757216464/
    └── session_757216464.nwb
```

## Installation

This project was developed with Python 3.11.

Install the required packages with:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/explorations.ipynb
```

## Planned Experiments

* Measure decoding accuracy as the number of sampled VISp neurons increases.
* Repeat random neuron sampling to estimate variability at each population size.
* Collapse opposite directions into four orientation classes.
* Compare eight-direction decoding with four-orientation decoding.
* Later, evaluate performance using cross-validation.

## Tools

* Python
* NumPy
* pandas
* h5py
* Matplotlib
* scikit-learn
* Jupyter Notebook

## Status

Work in progress. The current repository contains the completed baseline direction decoder. Additional population-size and direction-versus-orientation experiments will be added through later commits.
