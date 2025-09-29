# 1P1Qm_SF: Quantum Machine Learning for Jet Classification

This repo contains photonic quantum machine learning experiments built on Strawberry Fields (TensorFlow backend) for HEP jets. It includes:
- A binary classifier (`sf_refactor/sf_main.py`)
- An anomaly-detection autoencoder (`sf_refactor/sf_autoencoder.py`) with trash modes
- Multiple circuit families, including the interferometer–squeezing–interferometer architecture ("int-s-int"), mapped to the `new_entangled` circuit
- HPC job script examples to mass-generate and submit sweeps
- A quick-analysis notebook (`sandbox.ipynb`) for plotting from logs and saved model folders


## Environment

- Create and activate a Conda env (adjust name if you like):

```bash
conda create -n qml-env --file 1P1Qm_SF/requirements.txt
conda activate qml-env
```

- You must provide the data files yourself and point the config to them. See the Data section below.


## Configuration (Hydra)

Edit `sf_refactor/config.yaml` or override on the CLI. Key groups:
- model:
  - `dim_cutoff`: Fock cutoff
  - `wires`: number of modes
  - `photon_modes`: number of modes to read out (≤ wires)
  - `which_circuit`: `default`, `new`, `multiuploading`, `reuploading`, or `new_entangled` (int-s-int)
  - Circuit-specific settings: `particles_per_wire`, `particle_mapping`, `reuploads_per_wire`
- training: `epochs`, `learning_rate`, `batch_size`, etc.
- data: paths and counts
  - `data_dir`, `val_dir`, `test_dir`: set these to wherever you store the .h5 files locally
  - `train_jets`, `val_jets`, `test_jets` (validation/test are adjusted to be divisible by batch size)
- optimization: early stopping and LR schedule
- runtime and profiling:
  - `cli_test`: when true, command-line only (no files written), good for quick checks
  - `run_parent_dir`, `run_name`, `run_dir`: control where outputs go

Output base directories (documented defaults):
- Classifier runs: `sf_refactor/saved_models_sf/classifier`
- Autoencoder runs: `sf_refactor/saved_models_sf/autoencoder`

By default, runs are grouped by day (YYYY_MM_DD) then minute (HH_MM); if `run_parent_dir` is provided, runs go under that subfolder, and autoencoder runs default to a name like `dimX_wiresY_trashZ` when `run_name` is not set.


## Circuits overview

- default: original embedding + fixed entanglers + per-mode trainables
- new: as above but with trainable CX strengths between pairs of modes
- multiuploading: sequentially upload multiple particles per mode
- reuploading: re-encode the same particle multiple times per mode
- maximally_entangled: same as new but BS architecture is maximally entangling
- new_entangled (int-s-int): Interferometer – Squeezing – Interferometer
  - I recommend using the full-name “interferometer–squeezing–interferometer (int-s-int)” circuit for the autoencoder; it maps to `which_circuit: new_entangled` in the config. (Not tried it for the classifier)


## Data

- Data files are not in this repo. Download/store your datasets locally and set:
  - `data.data_dir`: training file path (.h5)
  - `data.val_dir`: validation file path (.h5)
  - `data.test_dir`: test file path (.h5)
- Labels: 0 = background, 1 = signal. For the autoencoder, training defaults to a single label (background) via `autoencoder.train_label`.


## Running: classifier (`sf_main.py`)

- Minimal quick run (CLI test; prints progress, writes no files):
```bash
cd sf_refactor
python sf_main.py runtime.cli_test=true runtime.run_name=test_run
```

- Regular run (typically won't use since you'll submit a job instead):
```bash
cd sf_refactor
nohup python sf_main.py \
  runtime.run_name=run_name \
  model.which_circuit="new_entangled" \
  model.dim_cutoff=9 \
  model.wires=5 \
  > sf_main_logs/prod_run.log 2>&1 &
```

Memory guidance: Fock simulation RAM grows exponentially in cutoff × wires. High-memory zones are the diagonal and above; prefer small configs for dev and consider `batch_size=1` when near limits.


## Running: autoencoder (`sf_autoencoder.py`)

- Recommended circuit: interferometer–squeezing–interferometer (int-s-int) → `which_circuit: new_entangled`

- Minimal quick run (CLI test):
```bash
cd sf_refactor
python sf_autoencoder.py \
  runtime.cli_test=true \
  model.which_circuit=new_entangled \
  model.dim_cutoff=3 \
  model.wires=5 \
  data.train_jets=50 \
  data.val_jets=50 \
  data.test_jets=50
```

- Regular run (again, typically submit a job instead):
```bash
cd sf_refactor
nohup python sf_autoencoder.py \
  model.which_circuit=new_entangled \
  model.dim_cutoff=6 \
  model.wires=4 \
  runtime.run_parent_dir=experiments/scanA \
  runtime.run_name=my_intsint_run \
  > sf_auto_logs/intsint_run.log 2>&1 &
```

Trash modes in the AE: training minimises the summed mean photon number on selected “trash” modes (driven toward vacuum). We compute anomaly scores from those modes and report AUC ("Final test AUC (anomaly score)").


## CLI test mode

- `runtime.cli_test=true` runs print-rich, command-line-only training for quick checks:
  - Prints epoch summaries, gradient norms, memory usage, etc
  - Does not write run directories or files (no params/model/plots/logs)


## What gets saved in a run directory (non-CLI test)

- params.txt: Config params (+ seed, timestamps)
- gradients.txt: per-epoch average gradient norms
- memory_profile.txt: periodic RAM (and GPU if available) logs
- model/:
  - model_weights.pkl: trained parameters
  - model_config.yaml: resolved OmegaConf snapshot
  - feature_scaling.pkl: feature scaling used during training
  - circuit_architecture.pkl: circuit metadata (type, wires, cutoff, etc.)
  - metadata.pkl: framework/version metadata
- plots/:
  - roc_curve.png
  - score_histogram.png
  - mean_photon_evolution.png (autoencoder only)
- test_predictions.txt: CSV-like predictions with inputs
- autoencoder only:
  - anomaly_scores.txt
  - mean_photon_training_log.csv (per-step mean-photon on trash modes)
  - fock_diagnostics.jsonl (periodic snapshots if enabled)


## HPC job scripts: examples for mass sweeps

- Generate autoencoder jobs across cutoff×wires with GPU vs CPU split based on memory heuristics:
  - `sf_refactor/generate_autoencoder_jobs.sh` creates `.pbs` files into `sf_refactor/experiment_jobs_autoencoder/`
  - Logs are written to `sf_refactor/sf_auto_logs/dim_vs_wire_table/` by default (example path; actual output depends on your scheduler job config)
- Submit them:
  - `sf_refactor/submit_all_autoencoder_jobs.sh` lets you submit all, or filter GPU-only (ngpus=1) vs high-memory CPU-only (mem≈4TB)
- Activate environment inside jobs via `conda activate qml-env`
- GPU vs high-memory boundary: most feasible runs are on GPU up to roughly cutoff 8 with moderate wires; diagonal and top-right (e.g., dim≥9 & wires≥5) require high-memory nodes. Few practical configs exist between “GPU ok” and “too high-memory”.


## Notebook: quick plotting/analysis

- `sandbox.ipynb` is for ad-hoc, multi-file plotting and inspection of saved runs and logs (both classifier and autoencoder). Typical usage:
  - Load `params.txt`, `test_predictions.txt`, `anomaly_scores.txt`, plots, and logs from a run folder
  - Produce custom figures, compare multiple runs, and aggregate AUCs


## Analysis scripts

- `sf_refactor/collect_results.py`: tabulate classifier AUCs for cutoff×wires sweeps (reads `sf_main_logs/dim_vs_wire_table`).
- `sf_refactor/collect_auto_results.py`: same for autoencoder (reads `sf_auto_logs/dim_vs_wire_table`).


## Inference (might be outdated)

You can load a saved model for simple scoring with `simple_inference.py` or by reading `model/` artifacts directly. Example (paths illustrative):
```bash
cd sf_refactor
python simple_inference.py --model_dir saved_models_sf/.../model
```


## NB

- Memory grows fast with `dim_cutoff` and `wires`; for testing, use small values and consider `batch_size=1` near the limits.
- Validation/test sizes are rounded down to a multiple of `batch_size` automatically.
- When using `multiuploading`, ensure `wires × particles_per_wire` ≤ available particles per jet in your dataset, otherwise a clear error is raised.

