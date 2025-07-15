# 1P1Qm_SF: Quantum Machine Learning for Jet Classification

This repository implements a quantum machine learning model using Strawberry Fields for jet classification in high-energy physics.

A photonic quantum circuit encodes jet features (η, φ, pT) for binary classification.

## Setup

Use Miniconda or your preferred variant.

1. Create and activate environment:
   ```bash
   conda create --name qml-env --file requirements.txt
   conda activate qml-env
   ```

2. Edit `sf_refactor/config.yaml` for parameters using Hydra:
   - `model.dim_cutoff`: Fock cutoff (memory scales exponentially)
   - `model.wires`: Number of wires/qubits (memory scales exponentially)
   - `training.epochs`, `training.batch_size`, etc.
   - Data paths: `data.data_dir`, `data.val_dir`, `data.test_dir`
   - Can also edit in cli 

## Running

- **Quick test (CLI mode) - Use `nohup` to save output to a log file in `sf_refactor/sf_main_logs/`:**
  ```bash
  cd sf_refactor
  nohup python sf_main.py runtime.cli_test=true runtime.run_name=test_run > sf_main_logs/test_run.log 2>&1 &
  ```
- **Regular run (slow, hours):**
    ```bash
    nohup python sf_main.py \
        runtime.run_name=run_name \
        model.dim_cutoff=9 \
        model.wires=5 \
        data.train_jets=1000 \
        data.val_jets=200 \
        data.test_jets=1000 \
        > sf_main_logs/prod_run.log 2>&1 &
    ```

## HPC Job Submission

- **Submit job:**
    ```bash
    qsub high_memory.pbs
    ```

- **View job status:**
    ```bash
    qstat -u $USER
    tail -f /path/to/job/output.log   # (Note: 'tail -f' may not work reliably on some HPC systems)
    ```

## Output

- Model outputs: `saved_models_sf/<run_name>/`
  - `params.txt`: Configuration
  - `gradients.txt`: Gradient norms
  - `plots/`: ROC curve, score histogram

## Memory Warning

- Memory usage grows exponentially with `dim_cutoff` and `wires`:

    | dim | wires | Required Memory  |
    | --- | ----- | ---------------- |
    | 10  | 6     | 8 TB             |
    | 10  | 5     | **80 GB**        |
    | 9   | 6     | **2.3 TB**       |
    | 8   | 6     | **549 GB**       |

- Use small values for development.
- Use `batch_size=1` for high-memory configs.
- Do not exceed `dim_cutoff=10` unless you have sufficient resources.
- Use `wires=5` or fewer for most runs.

