import os
import random
from typing import List

import numpy as np
import strawberryfields as sf
import tensorflow as tf
from omegaconf import DictConfig
import hydra
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from helpers.weights import (
    CircuitKind,
    make_symbolic_and_circuit,
    make_trainables,
    build_runtime_args,
)
from helpers.config import (
    print_config,
    save_config_to_file,
    setup_run_name,
    create_run_directory,
    validate_and_adjust_config,
)
from helpers.memory_profiler import create_memory_profiler
from helpers.model_utils import save_quantum_model
from helpers.plotting import plot_roc_curve, plot_score_histogram, plot_mean_photon_evolution
from helpers.predictions import save_test_predictions, save_anomaly_scores
from helpers.utils import load_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


# --- Force Randomness ---
seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def _get_cfg_trash_modes(cfg) -> List[int]:
    # Fallback to [0,1] if not present in config
    try:
        trash_modes = list(cfg.autoencoder.trash_modes)
    except Exception:
        trash_modes = [0, 1]
    return trash_modes


def _get_train_label(cfg) -> int:
    try:
        return int(cfg.autoencoder.train_label)
    except Exception:
        return 0  # background by default


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Validate and adjust configuration
    cfg = validate_and_adjust_config(cfg)

    # Setup run directory (autoencoder) with day grouping and time-only run name
    run_name = create_run_directory(cfg, "autoencoder")

    # Save configuration to file (exclude classifier loss fields for autoencoder)
    save_config_to_file(cfg, run_name, seed, exclude_loss=True)

    # Print configuration (hide loss line for autoencoder)
    print_config(cfg, run_name, seed, hide_loss=True)

    # Initialize memory profiler
    profiler = create_memory_profiler(cfg, run_name)
    if profiler:
        profiler.log_memory("Script start")

    # ----------  load datasets ----------
    # Train on background only (or specified train_label)
    train_label = _get_train_label(cfg)
    if profiler:
        with profiler.profile_block("Data loading"):
            jets_bg, labels_bg, jet_pt_bg = load_data(
                cfg.data.data_dir,
                max_jets=cfg.data.train_jets,
                num_particles=cfg.model.num_particles_needed,
                filter_label=train_label,
            )
            jets_val, labels_val, jet_pt_val = load_data(
                cfg.data.val_dir,
                max_jets=cfg.data.val_jets,
                num_particles=cfg.model.num_particles_needed,
            )
            jets_test, labels_test, jet_pt_test = load_data(
                cfg.data.test_dir,
                max_jets=cfg.data.test_jets,
                num_particles=cfg.model.num_particles_needed,
            )
    else:
        jets_bg, labels_bg, jet_pt_bg = load_data(
            cfg.data.data_dir,
            max_jets=cfg.data.train_jets,
            num_particles=cfg.model.num_particles_needed,
            filter_label=train_label,
        )
        jets_val, labels_val, jet_pt_val = load_data(
            cfg.data.val_dir,
            max_jets=cfg.data.val_jets,
            num_particles=cfg.model.num_particles_needed,
        )
        jets_test, labels_test, jet_pt_test = load_data(
            cfg.data.test_dir,
            max_jets=cfg.data.test_jets,
            num_particles=cfg.model.num_particles_needed,
        )

    # -------- symbolic circuit ----------
    prog = sf.Program(cfg.model.wires)

    # Circuit selection via config; support new_entangled explicitly
    circuit_selector = getattr(cfg.model, "which_circuit", None)
    which = None
    if isinstance(circuit_selector, str):
        key = circuit_selector.lower()
        if key in {"maximally_entangled", "max-entangled", "max_entangled", "clements_50_50", "maxent"} or bool(getattr(cfg.model, "use_maximally_entangled", False)):
            which = CircuitKind.MAX_ENT
        elif key in {"new_entangled", "new-entangled", "interferometer_squeeze_interferometer", "int-s-int"}:
            which = CircuitKind.NEW_ENT
        else:
            which = CircuitKind.NEW
    else:
        which = CircuitKind.NEW

    import circuits as circuits_module
    sym, circuit_fn = make_symbolic_and_circuit(prog, cfg.model.wires, which, circuits_module=circuits_module)
    prog = circuit_fn(prog, cfg.model.wires, sym.as_weights_dict(cfg.model.wires))

    # Trainable variables per circuit kind
    trainables = make_trainables(cfg.model.wires, which)

    # -------- Initialise variables ----------
    # Trainables created above in trainables

    def make_args(jet_batch, jet_pt_batch):
        return build_runtime_args(cfg.model.wires, jet_batch, jet_pt_batch, sym, trainables)

    opt = tf.keras.optimizers.Adam(cfg.training.learning_rate)

    # SF batch_size cannot be 1 -- instead use None
    sf_batch_size = cfg.training.batch_size if cfg.training.batch_size != 1 else None

    # Initialize StrawberryFields engine
    if profiler:
        with profiler.profile_block("Engine initialization"):
            eng = sf.Engine(
                "tf",
                backend_options={
                    "cutoff_dim": cfg.model.dim_cutoff,
                    "batch_size": sf_batch_size,
                },
            )
    else:
        eng = sf.Engine(
            "tf", backend_options={"cutoff_dim": cfg.model.dim_cutoff, "batch_size": sf_batch_size}
        )

    # -------- training loop ----------
    print("Starting autoencoder training (background-only)...", flush=True)

    # Early stopping and LR tracking (use AUC on validation as guidance; score = P0)
    best_val_auc = 0.0
    patience_counter = 0
    lr_patience_counter = 0
    best_weights = None
    current_lr = cfg.training.learning_rate

    # Train dataset uses background-only jets
    train_dataset = tf.data.Dataset.from_tensor_slices((jets_bg, jet_pt_bg))
    train_dataset = train_dataset.shuffle(buffer_size=jets_bg.shape[0]).batch(
        cfg.training.batch_size, drop_remainder=True
    )

    var_names = trainables.var_names(cfg.model.wires)

    trash_modes = _get_cfg_trash_modes(cfg)
    if max(trash_modes) >= cfg.model.wires:
        raise ValueError(
            f"Trash modes {trash_modes} exceed number of wires {cfg.model.wires}."
        )

    # Tracking containers for per-step mean photon numbers on trash modes
    mean_photon_steps = {m: [] for m in trash_modes}
    mean_photon_values = {m: [] for m in trash_modes}
    global_step = 0

    for epoch in range(cfg.training.epochs):
        if profiler:
            profiler.log_memory(f"Epoch {epoch+1} start")

        epoch_loss = 0.0
        num_steps = 0
        epoch_grad_norms = {}

        train_iter = (
            tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
            if cfg.runtime.cli_test
            else enumerate(train_dataset)
        )

        for step, (jet_batch, jet_pt_batch) in train_iter:
            if eng.run_progs:
                eng.reset()

            with tf.GradientTape() as tape:
                state = eng.run(prog, args=make_args(jet_batch, jet_pt_batch)).state
                # Sum mean photon on selected trash modes, and log per-mode values
                p0_list = [state.mean_photon(m)[0] for m in trash_modes]
                # Record per-mode mean photon values by global training step
                if not cfg.runtime.cli_test:
                    for m, val in zip(trash_modes, p0_list):
                        mean_photon_steps[m].append(global_step)
                        # val may be batched (shape [batch]); log batch-mean scalar
                        try:
                            val_mean = tf.reduce_mean(val)
                            mean_photon_values[m].append(float(val_mean.numpy()))
                        except Exception:
                            # Fallback: best-effort scalar conversion
                            mean_photon_values[m].append(float(np.mean(val.numpy())))
                if cfg.training.batch_size == 1:
                    p0 = tf.expand_dims(tf.add_n(p0_list), axis=0)
                else:
                    p0 = tf.add_n(p0_list)
                # loss = mean over batch of P0
                loss = tf.reduce_mean(p0)

            vars_ = trainables.list_vars()
            grads = tape.gradient(loss, vars_)
            opt.apply_gradients(zip(grads, vars_))

            epoch_loss += float(loss.numpy())
            num_steps += 1

            for i, g in enumerate(grads):
                if g is not None:
                    name = var_names[i]
                    epoch_grad_norms.setdefault(name, []).append(float(tf.norm(g).numpy()))

            if cfg.runtime.cli_test:
                train_iter.set_postfix(loss=f"{loss.numpy():.4f}")

            if profiler and cfg.profiling.memory_log_frequency > 0 and (step + 1) % cfg.profiling.memory_log_frequency == 0:
                profiler.log_memory(
                    f"Epoch {epoch+1}, batch {step+1}/{len(train_dataset)}"
                )
            if not cfg.runtime.cli_test and (step + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{cfg.training.epochs}, Step {step+1}/{len(train_dataset)} - Loss: {loss:.4e}", flush=True)
            global_step += 1
        
        avg_train_loss = epoch_loss / max(1, num_steps)

        # -------- validation: compute anomaly scores on mixed val set and AUC --------
        val_scores = []
        num_val_batches = cfg.data.val_jets // cfg.training.batch_size
        for i in range(num_val_batches):
            start, end = i * cfg.training.batch_size, (i + 1) * cfg.training.batch_size
            jet_batch_val = jets_val[start:end]
            jet_pt_batch_val = jet_pt_val[start:end]

            if eng.run_progs:
                eng.reset()
            state = eng.run(prog, args=make_args(jet_batch_val, jet_pt_batch_val)).state
            p0_list = [state.mean_photon(m)[0] for m in trash_modes]
            if cfg.training.batch_size == 1:
                p0 = tf.expand_dims(tf.add_n(p0_list), axis=0)
            else:
                p0 = tf.add_n(p0_list)
            val_scores.extend(p0.numpy())

        # Higher score => more anomalous (signal). Compute AUC against label==1.
        auc_val = roc_auc_score(labels_val.numpy(), np.asarray(val_scores))

        # Early stopping / LR schedule based on AUC with margin
        if auc_val > best_val_auc + cfg.optimization.min_delta:
            best_val_auc = auc_val
            patience_counter = 0
            lr_patience_counter = 0

            if cfg.optimization.restore_best:
                best_weights = trainables.export_numpy(cfg.model.wires)
        else:
            patience_counter += 1
            lr_patience_counter += 1
            if (
                lr_patience_counter >= cfg.optimization.lr_patience
                and current_lr > cfg.optimization.min_lr
            ):
                old_lr = current_lr
                current_lr = max(
                    current_lr * cfg.optimization.lr_factor, cfg.optimization.min_lr
                )
                opt.learning_rate.assign(current_lr)
                lr_patience_counter = 0
                print(f"Reduced learning rate from {old_lr} to {current_lr}", flush=True)

        print(
            f"Epoch {epoch+1}/{cfg.training.epochs} - Train P0: {avg_train_loss:.4f} - Val AUC: {auc_val:.4f}",
            flush=True,
        )

        min_epochs = getattr(cfg.optimization, "min_epochs", 0)
        if patience_counter >= cfg.optimization.patience and epoch + 1 >= min_epochs:
            print(
                f"Early stopping triggered after {epoch+1} epochs (minimum: {min_epochs})",
                flush=True,
            )
            if cfg.optimization.restore_best and best_weights is not None:
                # Assign back to variables using names mapping
                name_to_var = {n: v for n, v in zip(trainables.var_names(cfg.model.wires), trainables.list_vars())}
                for k, val in best_weights.items():
                    if k in name_to_var:
                        name_to_var[k].assign(val)
                print(f"Restored best weights with Val AUC: {best_val_auc:.4f}", flush=True)
            break
        elif patience_counter >= cfg.optimization.patience:
            print(
                f"Early stopping criteria met but continuing (epoch {epoch+1} < minimum {min_epochs})",
                flush=True,
            )

        if profiler:
            profiler.log_memory(f"Epoch {epoch+1} end")

    # -------- Save model (weights only, no bias here) --------
    if not cfg.runtime.cli_test:
        print("Saving trained autoencoder model...", flush=True)
        model_weights = trainables.export_numpy(cfg.model.wires)
        model_dir = save_quantum_model(model_weights, cfg, run_name, cfg.data.save_dir)

    # -------- Evaluate on test set: anomaly scores and AUC --------
    def compute_scores(jets_tensor, jet_pt_tensor):
        scores = []
        total = jets_tensor.shape[0]
        num_batches = total // cfg.training.batch_size
        for i in range(num_batches):
            start = i * cfg.training.batch_size
            end = start + cfg.training.batch_size
            jet_batch = jets_tensor[start:end]
            jet_pt_batch = jet_pt_tensor[start:end]

            if eng.run_progs:
                eng.reset()
            state = eng.run(prog, args=make_args(jet_batch, jet_pt_batch)).state
            p0_list = [state.mean_photon(m)[0] for m in trash_modes]
            if cfg.training.batch_size == 1:
                p0 = tf.expand_dims(tf.add_n(p0_list), axis=0)
            else:
                p0 = tf.add_n(p0_list)
            scores.extend(p0.numpy())

            if (end) >= total or (i + 1) % 5 == 0:
                print(f"  Processed {min(end, total)}/{total} jets", flush=True)
        return np.asarray(scores)

    if profiler:
        with profiler.profile_block("Test evaluation"):
            print("Scoring test set...", flush=True)
            score_test = compute_scores(jets_test, jet_pt_test)
            auc_test = roc_auc_score(labels_test.numpy(), score_test)
    else:
        print("Scoring test set...", flush=True)
        score_test = compute_scores(jets_test, jet_pt_test)
        auc_test = roc_auc_score(labels_test.numpy(), score_test)

    # Optionally convert scores to a binary prediction with a default threshold
    pred_test = (score_test >= np.median(score_test)).astype(int)
    accuracy_test = np.mean(pred_test == labels_test.numpy())

    print("Training completed.", flush=True)
    print(f"Final test AUC (anomaly score): {auc_test:.4f}", flush=True)
    print(f"Final test Accuracy (median threshold): {accuracy_test:.4f}", flush=True)

    if not cfg.runtime.cli_test:
        # Save predictions (reuse existing saver + extra anomaly scores file)
        save_test_predictions(
            cfg, run_name, score_test, pred_test, labels_test, jet_pt_test, jets_test
        )
        save_anomaly_scores(cfg, run_name, score_test)
        # Generate and save plots (ROC and score histogram)
        plots_dir = os.path.join(cfg.runtime.run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
        score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
        plot_roc_curve(labels_test.numpy(), score_test, roc_plot_path)
        plot_score_histogram(labels_test.numpy(), score_test, score_hist_path)
        print(f"Plots saved to {plots_dir}")
        # Save mean photon logs and plot their evolution
        # Write CSV with columns: step, mode, mean_photon
        mp_csv = os.path.join(cfg.runtime.run_dir, "mean_photon_training_log.csv")
        with open(mp_csv, "w") as f:
            f.write("step,mode,mean_photon\n")
            for m in trash_modes:
                for s, v in zip(mean_photon_steps[m], mean_photon_values[m]):
                    f.write(f"{s},{m},{v}\n")
        mp_plot = os.path.join(plots_dir, 'mean_photon_evolution.png')
        plot_mean_photon_evolution(mean_photon_steps, mean_photon_values, mp_plot)
        print(f"Mean photon log saved to {mp_csv}; plot saved to {mp_plot}")

    if profiler:
        profiler.log_memory("Script end")


if __name__ == "__main__":
    main()
