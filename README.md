# Topological signatures of human activity

One row of the UCI HAR data is a 2.56-second sensor window from one person doing one of six activities. The project turns each window into separate acceleration and angular-velocity delay clouds, extracts persistent topology and a cocycle-derived circular phase, then studies Euclidean and max-product coupled sensor filtrations. It asks five questions:

1. Do their stable ranks contain activity information that transfers to new people?
2. Does the information come from time order rather than the unordered sensor values?
3. Can the longest persistent loop define an interpretable motion phase across all six activities?
4. Do topology-and-phase features improve activity classification?
5. Do coupled acceleration--gyroscope filtrations add information beyond the two separate clouds?

The answer to the first four questions is **yes**, with an important qualification: the 16 topology-and-phase features reach 0.611 held-out balanced accuracy, but ordinary signal features remain substantially stronger. For the fifth, both product metrics add complementary transformations, but marginal-preserving controls show that correct sensor synchronization is not the source of the gain.

## Read the code in this order

| File | What happens there |
|---|---|
| `data_loader.py` | Loads windows, labels, subject IDs, and the supplied 561 features. |
| `embedding.py` | Builds separate $98\times9$ delay clouds and the synchronized $98\times18$ joint cloud. |
| `topology.py` | Normalizes clouds and computes separate, Euclidean-joint, and max-coupled $H_0/H_1$ barcodes. |
| `features.py` | Converts barcodes to stable ranks, diagram summaries, and circular phase features; also builds the ordinary baseline. |
| `evaluation.py` | Runs nested subject-wise validation and evaluates the official test subjects. |
| `run_analysis.py` | Defines the seven primary comparisons and saves their results. |
| `run_robustness.py` | Changes the embedding, metric, and input noise on training subjects. |
| `run_extensions.py` | Tests both coupled sensor metrics, broken sensor pairing, sensor/homology blocks, clustering, and the permutation null. |
| `run_dynamics.py` | Computes the same 16 interpretable topology-and-phase features for all six activities and links them to the final classifier. |

`analysis.py` is only an F5 entry point for Neovim. `analysis.ipynb` is the executed, plotted walkthrough; it reads cached all-window features rather than duplicating the expensive persistence calculations.

## Setup

Python 3.13 was used for the reported results. Create a local environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The raw [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) is not stored in this repository. `data_loader.py` downloads and safely extracts it into `data/` on first use.

## Reproduce the analysis

Run the scripts in this order:

```bash
python -m unittest -v test_analysis.py
python run_analysis.py
python run_robustness.py
python run_extensions.py
python run_dynamics.py
python make_figures.py
cd report
latexmk -pdf report.tex
```

Stable-rank matrices and fitted models are regenerated under the untracked `artifacts/` directory. Numerical results are in `results/`, plots are in `figures/`, and the finished paper is `report/report.pdf`. The primary test comparison was fixed before the test split was opened; analyses added afterwards are marked post-hoc in the report and protocol.

The coupled extensions are deliberately development-only. Euclidean-joint topology scores 0.438 alone and 0.560 when combined with the 0.533 separate topology. Max-coupled topology scores 0.425 alone and 0.557 when combined. In both cases, breaking acceleration--gyroscope pairing while preserving both marginal clouds performs at least as well after concatenation. The coupled vectors are complementary, but the evidence does not attribute that gain to correct synchronization.
