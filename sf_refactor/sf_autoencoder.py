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

from circuits import new_circuit
from helpers.config import (
    print_config,
    save_config_to_file,
    setup_run_name,
    validate_and_adjust_config,
)
from helpers.memory_profiler import create_memory_profiler
from helpers.model_utils import save_quantum_model
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

    # Setup run name
    run_name = setup_run_name(cfg)

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
    s_scale = prog.params("s_scale")
    disp_mag = [prog.params(f"disp_mag{w}") for w in range(cfg.model.wires)]
    disp_phase = [prog.params(f"disp_phase{w}") for w in range(cfg.model.wires)]
    squeeze_mag = [prog.params(f"squeeze_mag{w}") for w in range(cfg.model.wires)]
    squeeze_phase = [prog.params(f"squeeze_phase{w}") for w in range(cfg.model.wires)]
    eta = [prog.params(f"eta{w}") for w in range(cfg.model.wires)]
    phi = [prog.params(f"phi{w}") for w in range(cfg.model.wires)]
    pt = [prog.params(f"pt{w}") for w in range(cfg.model.wires)]

    # Trainable CX gate parameters
    cx_pairs = [(i, j) for i in range(cfg.model.wires) for j in range(i + 1, cfg.model.wires)]
    cx_theta = {(a, b): prog.params(f"cx_theta_{a}_{b}") for (a, b) in cx_pairs}

    # Use the base classifier circuit (new_circuit)
    weights = {
        "s_scale": s_scale,
        **{f"disp_mag_{w}": disp_mag[w] for w in range(cfg.model.wires)},
        **{f"disp_phase_{w}": disp_phase[w] for w in range(cfg.model.wires)},
        **{f"squeeze_mag_{w}": squeeze_mag[w] for w in range(cfg.model.wires)},
        **{f"squeeze_phase_{w}": squeeze_phase[w] for w in range(cfg.model.wires)},
        **{f"eta_{w}": eta[w] for w in range(cfg.model.wires)},
        **{f"phi_{w}": phi[w] for w in range(cfg.model.wires)},
        **{f"pt_{w}": pt[w] for w in range(cfg.model.wires)},
        **{f"cx_theta_{a}_{b}": cx_theta[(a, b)] for (a, b) in cx_pairs},
    }
    prog = new_circuit(prog, cfg.model.wires, weights)

    # -------- Initialise variables ----------
    rnd = tf.random_uniform_initializer(-0.1, 0.1)
    tf_s_scale = tf.Variable(rnd(()))
    tf_disp_mag = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_disp_phase = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_squeeze_mag = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_squeeze_phase = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_cx_theta = {(a, b): tf.Variable(rnd(())) for (a, b) in cx_pairs}

    # -------- Feature scaling ----------
    assumed_limits = {
        "pt": [1e-4, 3000.0],
        "eta": [-0.8, 0.8],
        "phi": [-0.8, 0.8],
    }
    feature_limits = {
        "pt": [0.0, 1.0],
        "eta": [-np.pi, np.pi],
        "phi": [-np.pi, np.pi],
    }

    def scale_feature(value, name):
        a_min, a_max = assumed_limits[name]
        f_min, f_max = feature_limits[name]
        return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

    def scale_pt_by_jet(particle_pts, jet_pt):
        return particle_pts / jet_pt[:, np.newaxis]

    def make_args(jet_batch, jet_pt_batch):
        squeeze_batch = jet_batch.shape[0] == 1

        d = {"s_scale": tf_s_scale}
        for w in range(cfg.model.wires):
            d[f"disp_mag{w}"] = tf_disp_mag[w]
            d[f"disp_phase{w}"] = tf_disp_phase[w]
            d[f"squeeze_mag{w}"] = tf_squeeze_mag[w]
            d[f"squeeze_phase{w}"] = tf_squeeze_phase[w]

        for w in range(cfg.model.wires):
            eta_val = scale_feature(jet_batch[:, w, 0], "eta")
            phi_val = scale_feature(jet_batch[:, w, 1], "phi")
            pt_val = scale_pt_by_jet(jet_batch[:, w, 2], jet_pt_batch)

            if squeeze_batch:
                d[f"eta{w}"] = tf.squeeze(eta_val)
                d[f"phi{w}"] = tf.squeeze(phi_val)
                d[f"pt{w}"] = tf.squeeze(pt_val)
            else:
                d[f"eta{w}"] = eta_val
                d[f"phi{w}"] = phi_val
                d[f"pt{w}"] = pt_val

        for (a, b) in cx_pairs:
            d[f"cx_theta_{a}_{b}"] = tf_cx_theta[(a, b)]
        return d

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

    var_names = [
        "s_scale",
        *[f"disp_mag_{i}" for i in range(cfg.model.wires)],
        *[f"disp_phase_{i}" for i in range(cfg.model.wires)],
        *[f"squeeze_mag_{i}" for i in range(cfg.model.wires)],
        *[f"squeeze_phase_{i}" for i in range(cfg.model.wires)],
        *[f"cx_theta_{a}_{b}" for a, b in cx_pairs],
    ]

    trash_modes = _get_cfg_trash_modes(cfg)
    if max(trash_modes) >= cfg.model.wires:
        raise ValueError(
            f"Trash modes {trash_modes} exceed number of wires {cfg.model.wires}."
        )

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
                # Sum mean photon on selected trash modes
                p0_list = [state.mean_photon(m)[0] for m in trash_modes]
                if cfg.training.batch_size == 1:
                    p0 = tf.expand_dims(tf.add_n(p0_list), axis=0)
                else:
                    p0 = tf.add_n(p0_list)
                # loss = mean over batch of P0
                loss = tf.reduce_mean(p0)

            vars_ = [
                tf_s_scale,
                *tf_disp_mag,
                *tf_disp_phase,
                *tf_squeeze_mag,
                *tf_squeeze_phase,
                *tf_cx_theta.values(),
            ]
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
                best_weights = [
                    tf_s_scale.numpy(),
                    *[v.numpy() for v in tf_disp_mag],
                    *[v.numpy() for v in tf_disp_phase],
                    *[v.numpy() for v in tf_squeeze_mag],
                    *[v.numpy() for v in tf_squeeze_phase],
                    *[v.numpy() for v in tf_cx_theta.values()],
                ]
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
                vars_ = [
                    tf_s_scale,
                    *tf_disp_mag,
                    *tf_disp_phase,
                    *tf_squeeze_mag,
                    *tf_squeeze_phase,
                    *tf_cx_theta.values(),
                ]
                for var, w in zip(vars_, best_weights):
                    var.assign(w)
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
        model_weights = {"s_scale": tf_s_scale.numpy()}
        for w in range(cfg.model.wires):
            model_weights[f"disp_mag_{w}"] = tf_disp_mag[w].numpy()
            model_weights[f"disp_phase_{w}"] = tf_disp_phase[w].numpy()
            model_weights[f"squeeze_mag_{w}"] = tf_squeeze_mag[w].numpy()
            model_weights[f"squeeze_phase_{w}"] = tf_squeeze_phase[w].numpy()
        for (a, b) in cx_pairs:
            model_weights[f"cx_theta_{a}_{b}"] = tf_cx_theta[(a, b)].numpy()
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

    if profiler:
        profiler.log_memory("Script end")


if __name__ == "__main__":
    main()
