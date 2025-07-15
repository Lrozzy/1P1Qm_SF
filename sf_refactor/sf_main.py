import strawberryfields as sf
import tensorflow as tf
import numpy as np
import os, random
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from helpers.plotting import * 
from circuits import default_circuit, new_circuit
from helpers.utils import load_data, get_loss_fn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Force Randomness ---
# Set seeds for reproducibility. 
seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def validate_and_adjust_config(cfg: DictConfig) -> DictConfig:
    """Validate and adjust configuration parameters"""
    # Ensure photon_modes doesn't exceed wires
    cfg.model.photon_modes = min(cfg.model.photon_modes, cfg.model.wires)
    
    # Adjust val_jets and test_jets to be divisible by batch_size
    original_val_jets = cfg.data.val_jets
    cfg.data.val_jets = ((cfg.data.val_jets + cfg.training.batch_size - 1) // cfg.training.batch_size) * cfg.training.batch_size
    if original_val_jets != cfg.data.val_jets:
        print(f"Adjusting val_jets from {original_val_jets} to {cfg.data.val_jets} to be divisible by batch_size {cfg.training.batch_size}", flush=True)

    original_test_jets = cfg.data.test_jets
    cfg.data.test_jets = ((cfg.data.test_jets + cfg.training.batch_size - 1) // cfg.training.batch_size) * cfg.training.batch_size
    if original_test_jets != cfg.data.test_jets:
        print(f"Adjusting test_jets from {original_test_jets} to {cfg.data.test_jets} to be divisible by batch_size {cfg.training.batch_size}", flush=True)
    
    return cfg

def setup_run_name(cfg: DictConfig) -> str:
    """Setup unique run name with datetime and handle existing directories"""
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    if cfg.runtime.run_name:
        base_run_name = f"{now_str}_{cfg.runtime.run_name}"
    else:
        base_run_name = now_str
    
    run_name = base_run_name
    i = 1
    while os.path.exists(f'{cfg.data.save_dir}/{run_name}'):
        run_name = f"{base_run_name}_{i}"
        i += 1
    
    return run_name

def save_config_to_file(cfg: DictConfig, run_name: str, seed: int):
    """Save configuration parameters to a file"""
    if not cfg.runtime.cli_test:
        os.makedirs(os.path.join(cfg.data.save_dir, run_name), exist_ok=True)
        
        # Get current date and time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert config to dict and add runtime info
        params = OmegaConf.to_container(cfg, resolve=True)
        params['run_start_time'] = now
        params['seed'] = seed
        params['run_name'] = run_name
        
        # Flatten the nested config for easier reading
        flat_params = {}
        for section, values in params.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_params[f"{section}.{key}"] = value
            else:
                flat_params[section] = values
        
        params_path = os.path.join(cfg.data.save_dir, run_name, "params.txt")
        with open(params_path, "w") as f:
            for k, v in flat_params.items():
                f.write(f"{k}: {v}\n")

def print_config(cfg: DictConfig, run_name: str, seed: int):
    """Print configuration parameters"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Run started at: {now}", flush=True)
    print("PARAMETERS:")
    print(f"Random seed for this run: {seed}", flush=True)
    print(f"Dimension cutoff: {cfg.model.dim_cutoff}", flush=True)
    print(f"Wires: {cfg.model.wires}", flush=True)
    print(f"Layers: {cfg.model.layers}", flush=True)
    print(f"Epochs: {cfg.training.epochs}", flush=True)
    print(f"Learning rate: {cfg.training.learning_rate}", flush=True)
    print(f"Batch size: {cfg.training.batch_size}", flush=True)
    print(f"Loss function: {cfg.training.loss_fn}", flush=True)
    if cfg.training.tanh:
        print("\t Tanh activation: True", flush=True)
    else:
        print("\t Sigmoid activation with trainable bias", flush=True)
    print(f"Which circuit: {cfg.model.which_circuit}", flush=True)
    print(f"Train jets: {cfg.data.train_jets}", flush=True)
    print(f"Validation jets: {cfg.data.val_jets}", flush=True)
    print(f"Test jets: {cfg.data.test_jets}", flush=True)
    print(f"Photon modes: {cfg.model.photon_modes}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print("------------------------------------", flush=True)

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Validate and adjust configuration
    cfg = validate_and_adjust_config(cfg)
    
    # Setup run name
    run_name = setup_run_name(cfg)
    
    # Save configuration to file
    save_config_to_file(cfg, run_name, seed)
    
    # Print configuration
    print_config(cfg, run_name, seed)    # ----------  load datasets ---------- 
    jets, labels = load_data(cfg.data.data_dir, max_jets=cfg.data.train_jets, wires=cfg.model.wires)
    jets_val, labels_val = load_data(cfg.data.val_dir, max_jets=cfg.data.val_jets, wires=cfg.model.wires)
    jets_test, labels_test = load_data(cfg.data.test_dir, max_jets=cfg.data.test_jets, wires=cfg.model.wires)

    # -------- symbolic circuit ----------
    # print("Starting symbolic circuit construction...", flush=True)
    prog = sf.Program(cfg.model.wires)
    s_scale = prog.params("s_scale")
    disp_mag  = [prog.params(f"disp_mag{w}") for w in range(cfg.model.wires)] # disp_mag = Displacement Magnitude
    disp_phase  = [prog.params(f"disp_phase{w}") for w in range(cfg.model.wires)] # disp_phase = Displacement Phase
    squeeze_mag  = [prog.params(f"squeeze_mag{w}") for w in range(cfg.model.wires)] # squeeze_mag = Squeezing Magnitude
    squeeze_phase  = [prog.params(f"squeeze_phase{w}") for w in range(cfg.model.wires)] # squeeze_phase = Squeezing Phase
    eta = [prog.params(f"eta{w}") for w in range(cfg.model.wires)]
    phi = [prog.params(f"phi{w}") for w in range(cfg.model.wires)]
    pt  = [prog.params(f"pt{w}")  for w in range(cfg.model.wires)]

    # Extra variables for trainable cx gates
    cx_pairs = [(i, j) for i in range(cfg.model.wires) for j in range(i+1, cfg.model.wires)]
    cx_theta = {(a,b): prog.params(f"cx_theta_{a}_{b}") for (a,b) in cx_pairs}

    # -------- Circuit architecture ----------
    if cfg.model.which_circuit == "new":
        # weights is misleading since pt, eta, phi are included, consider changing
        weights = {
            's_scale': s_scale,
            **{f'disp_mag_{w}': disp_mag[w] for w in range(cfg.model.wires)},
            **{f'disp_phase_{w}': disp_phase[w] for w in range(cfg.model.wires)},
            **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(cfg.model.wires)},
            **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(cfg.model.wires)},
            **{f'eta_{w}': eta[w] for w in range(cfg.model.wires)},
            **{f'phi_{w}': phi[w] for w in range(cfg.model.wires)},
            **{f'pt_{w}': pt[w] for w in range(cfg.model.wires)},
            **{f"cx_theta_{a}_{b}": cx_theta[(a,b)] for (a,b) in cx_pairs},
        }
        prog = new_circuit(prog, cfg.model.wires, weights)
    else:
        weights = {
            's_scale': s_scale,
            **{f'disp_mag_{w}': disp_mag[w] for w in range(cfg.model.wires)},
            **{f'disp_phase_{w}': disp_phase[w] for w in range(cfg.model.wires)},
            **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(cfg.model.wires)},
            **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(cfg.model.wires)},
            **{f'eta_{w}': eta[w] for w in range(cfg.model.wires)},
            **{f'phi_{w}': phi[w] for w in range(cfg.model.wires)},
            **{f'pt_{w}': pt[w] for w in range(cfg.model.wires)},
        }
        prog = default_circuit(prog, cfg.model.wires, weights)

    # -------- Initialise variables ----------
    # print("Initialising variables...", flush=True)
    rnd = tf.random_uniform_initializer(-0.1, 0.1)
    tf_s_scale = tf.Variable(rnd(()))
    tf_disp_mag = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_disp_phase = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_squeeze_mag = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_squeeze_phase = [tf.Variable(rnd(())) for _ in range(cfg.model.wires)]
    tf_bias = tf.Variable(0.0, dtype=tf.float32) # Trainable bias

    # Extra variables for trainable cx gates
    tf_cx_theta = {(a,b): tf.Variable(rnd(())) for (a,b) in cx_pairs}    # -------- Feature scaling ----------
    # Define assumed limits for features
    assumed_limits = {
        'pt':  [1e-4, 3000.0],
        'eta': [-0.8, 0.8],
        'phi': [-0.8, 0.8],
    }
    # Define feature limits for scaling
    feature_limits = {
        'pt':  [0.0, 1.0],
        'eta': [-np.pi, np.pi],
        'phi': [-np.pi, np.pi],
    }

    def scale_feature(value, name):
        a_min, a_max = assumed_limits[name]
        f_min, f_max = feature_limits[name]
        return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

    # Create arguments for the SF program (map symbolic variables to tensors)
    def make_args(jet_batch):
        # This function now accepts a batch of jets
        
        # Squeeze the batch dimension if batch_size is 1
        squeeze = jet_batch.shape[0] == 1

        d = {"s_scale": tf_s_scale}
        for w in range(cfg.model.wires):
            d[f"disp_mag{w}"] = tf_disp_mag[w]
            d[f"disp_phase{w}"] = tf_disp_phase[w]
            d[f"squeeze_mag{w}"] = tf_squeeze_mag[w]
            d[f"squeeze_phase{w}"] = tf_squeeze_phase[w]
            
            # Slicing the batch to get all values for a specific feature across the batch
            eta_val = scale_feature(jet_batch[:, w, 0], "eta")
            phi_val = scale_feature(jet_batch[:, w, 1], "phi")
            pt_val  = scale_feature(jet_batch[:, w, 2], "pt")

            if squeeze:
                d[f"eta{w}"] = tf.squeeze(eta_val)
                d[f"phi{w}"] = tf.squeeze(phi_val)
                d[f"pt{w}"]  = tf.squeeze(pt_val)
            else:
                d[f"eta{w}"] = eta_val
                d[f"phi{w}"] = phi_val
                d[f"pt{w}"]  = pt_val

        if cfg.model.which_circuit == "new":
            for (a, b) in cx_pairs:
                d[f"cx_theta_{a}_{b}"] = tf_cx_theta[(a, b)]
        return d

    opt = tf.keras.optimizers.Adam(cfg.training.learning_rate)
    # print("Starting Engine...", flush=True)
    # SF batch_size cannot be 1 -- instead use None
    sf_batch_size = cfg.training.batch_size
    if cfg.training.batch_size == 1:
        sf_batch_size = None
    eng = sf.Engine("tf", backend_options={"cutoff_dim": cfg.model.dim_cutoff, "batch_size": sf_batch_size})    # -------- training loop ----------
    print("Starting training...", flush=True)

    # Early stopping and learning rate variables
    best_val_auc = 0.0
    patience_counter = 0
    lr_patience_counter = 0
    best_weights = None
    current_lr = cfg.training.learning_rate

    # Create a tf.data.Dataset for efficient batching and shuffling
    train_dataset = tf.data.Dataset.from_tensor_slices((jets, labels))
    train_dataset = train_dataset.shuffle(buffer_size=cfg.data.train_jets).batch(cfg.training.batch_size, drop_remainder=True)

    # Define var_names outside the loop
    if cfg.model.which_circuit == "new":
        var_names = ['s_scale'] + [f'disp_mag_{i}' for i in range(cfg.model.wires)] + [f'disp_phase_{i}' for i in range(cfg.model.wires)] + [f'squeeze_mag_{i}' for i in range(cfg.model.wires)] + [f'squeeze_phase_{i}' for i in range(cfg.model.wires)] + [f'cx_theta_{a}_{b}' for a,b in cx_pairs] + ['bias']
    else:
        var_names = ['s_scale'] + [f'disp_mag_{i}' for i in range(cfg.model.wires)] + [f'disp_phase_{i}' for i in range(cfg.model.wires)] + [f'squeeze_mag_{i}' for i in range(cfg.model.wires)] + [f'squeeze_phase_{i}' for i in range(cfg.model.wires)] + ['bias']

    for epoch in range(cfg.training.epochs):
        # Keep track of the loss for this epoch
        epoch_loss = 0.0
        num_steps = 0
        # Use a dictionary to store gradient norms for each variable
        epoch_grad_norms = {}
        
        # Loop over batches in the training set
        if cfg.runtime.cli_test:
            train_iterator = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        else:
            train_iterator = enumerate(train_dataset)

        for step, (jet_batch, label_batch) in train_iterator:
            if eng.run_progs:
                eng.reset()

            with tf.GradientTape() as tape:
                # Run the circuit for the entire batch
                state   = eng.run(prog, args=make_args(jet_batch)).state
                # state.mean_photon(m) returns a tuple (mean, variance), we only want the mean
                photons_list = [state.mean_photon(m)[0] for m in range(cfg.model.photon_modes)]
                if cfg.training.batch_size == 1:
                    photons = tf.expand_dims(tf.stack(photons_list, axis=0), axis=0)
                else:
                    photons = tf.stack(photons_list, axis=1)
                # Calculate loss for the batch
                loss_vector, logit_to_prob = get_loss_fn(photons, label_batch, bias=tf_bias, tanh=cfg.training.tanh, loss_type=cfg.training.loss_fn)
                # Average the loss over the batch for a stable gradient
                loss = tf.reduce_mean(loss_vector)

            if cfg.model.which_circuit == "new":
                vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, *tf_cx_theta.values(), tf_bias]
            else:
                vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, tf_bias]
            grads = tape.gradient(loss, vars_)
            opt.apply_gradients(zip(grads, vars_))
            
            epoch_loss += loss.numpy()
            num_steps += 1

            # Store gradient norms for each variable
            for i, g in enumerate(grads):
                if g is not None:
                    var_name = var_names[i]
                    if var_name not in epoch_grad_norms:
                        epoch_grad_norms[var_name] = []
                    epoch_grad_norms[var_name].append(tf.norm(g).numpy())

            if cfg.runtime.cli_test:
                train_iterator.set_postfix(loss=f"{loss.numpy():.4f}")

            if not cfg.runtime.cli_test and (step + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{cfg.training.epochs}, Step {step+1}/{len(train_dataset)} - Batch Loss: {loss:.4f}", flush=True)        # -------- validation step at the end of each epoch ----------
        avg_train_loss = epoch_loss / num_steps
        
        # Save gradients for the epoch
        if not cfg.runtime.cli_test:
            gradients_path = os.path.join(cfg.data.save_dir, run_name, "gradients.txt")
            with open(gradients_path, "a") as f:
                f.write(f"--- Epoch {epoch+1} ---\n")
                for name, norm_list in epoch_grad_norms.items():
                    if norm_list:
                        avg_norm = np.mean(norm_list)
                        f.write(f"{name}: {avg_norm}\n")
                f.write("\n")

        val_probs_pass = []
        val_losses_pass = []
        
        # Process validation set in batches
        num_val_batches = cfg.data.val_jets // cfg.training.batch_size
        for i in range(num_val_batches):
            start = i * cfg.training.batch_size
            end = start + cfg.training.batch_size
            jet_batch_val = jets_val[start:end]
            label_batch_val = labels_val[start:end]

            if eng.run_progs:
                eng.reset()

            state   = eng.run(prog, args=make_args(jet_batch_val)).state
            photons_list = [state.mean_photon(m)[0] for m in range(cfg.model.photon_modes)]
            if cfg.training.batch_size == 1:
                photons = tf.expand_dims(tf.stack(photons_list, axis=0), axis=0)
            else:
                photons = tf.stack(photons_list, axis=1)
            val_loss_vector, val_prob = get_loss_fn(photons, label_batch_val, bias=tf_bias, tanh=cfg.training.tanh, loss_type=cfg.training.loss_fn)
            
            val_probs_pass.extend(val_prob.numpy())
            val_losses_pass.extend(val_loss_vector.numpy())

        avg_val_loss = np.mean(val_losses_pass)
        auc_val = roc_auc_score(labels_val.numpy(), np.asarray(val_probs_pass))
        
        # Early stopping and learning rate logic
        if auc_val > best_val_auc + cfg.optimization.min_delta:
            best_val_auc = auc_val
            patience_counter = 0
            lr_patience_counter = 0
            
            # Save best weights
            if cfg.optimization.restore_best:
                if cfg.model.which_circuit == "new":
                    best_weights = [tf_s_scale.numpy(), *[v.numpy() for v in tf_disp_mag], 
                                    *[v.numpy() for v in tf_disp_phase], *[v.numpy() for v in tf_squeeze_mag],
                                    *[v.numpy() for v in tf_squeeze_phase], *[v.numpy() for v in tf_cx_theta.values()], 
                                    tf_bias.numpy()]
                else:
                    best_weights = [tf_s_scale.numpy(), *[v.numpy() for v in tf_disp_mag], 
                                    *[v.numpy() for v in tf_disp_phase], *[v.numpy() for v in tf_squeeze_mag],
                                    *[v.numpy() for v in tf_squeeze_phase], tf_bias.numpy()]
            
            # print(f"  New best validation AUC: {best_val_auc:.4f}", flush=True)
        else:
            patience_counter += 1
            
            # Learning rate reduction on plateau
            lr_patience_counter += 1
            if lr_patience_counter >= cfg.optimization.lr_patience and current_lr > cfg.optimization.min_lr:
                old_lr = current_lr
                current_lr = max(current_lr * cfg.optimization.lr_factor, cfg.optimization.min_lr)
                opt.learning_rate.assign(current_lr)
                lr_patience_counter = 0
                print(f"Reduced learning rate from {old_lr} to {current_lr}", flush=True)
            
            # print(f"  No improvement. Patience: {patience_counter}/{patience}", flush=True)
        
        # Early stopping check
        if patience_counter >= cfg.optimization.patience:
            print(f"Early stopping triggered after {epoch+1} epochs", flush=True)
            
            # Restore best weights
            if cfg.optimization.restore_best and best_weights is not None:
                if cfg.model.which_circuit == "new":
                    vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, *tf_cx_theta.values(), tf_bias]
                else:
                    vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, tf_bias]
                
                for var, weight in zip(vars_, best_weights):
                    var.assign(weight)
                print(f"Restored best weights from epoch with AUC: {best_val_auc:.4f}", flush=True)
            
            break
        
        if cfg.runtime.cli_test:
            print(f"Epoch {epoch+1}/{cfg.training.epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation AUC: {auc_val:.4f}", flush=True)
            # Diagnostic: Print the first 5 validation probabilities to check if they are changing
            val_probs_preview = ", ".join([f"{p:.4f}" for p in val_probs_pass[:5]])
            print(f"  Validation Probs Preview: [{val_probs_preview}]", flush=True)
            print("  Average Gradient Norms per variable for this epoch:", flush=True)
            for var_name, norms in sorted(epoch_grad_norms.items()):
                avg_norm = np.mean(norms) if norms else 0.0
                print(f"    {var_name}: {avg_norm:.6f}", flush=True)
        else:
            print(f"Epoch {epoch+1}/{cfg.training.epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation AUC: {auc_val:.4f}", flush=True)    # --------- Evaluate and print AUC ---------
    def predict_prob(jets_tensor, labels):
        """Return an array of P(signal) for each jet."""
        probs = []
        total_jets = jets_tensor.shape[0]
        num_batches = total_jets // cfg.training.batch_size

        for i in range(num_batches):
            start = i * cfg.training.batch_size
            end = start + cfg.training.batch_size
            jet_batch = jets_tensor[start:end]
            label_batch = labels[start:end]

            if eng.run_progs:
                eng.reset()
            state   = eng.run(prog, args=make_args(jet_batch)).state
            photons_list = [state.mean_photon(m)[0] for m in range(cfg.model.photon_modes)]
            if cfg.training.batch_size == 1:
                photons = tf.expand_dims(tf.stack(photons_list, axis=0), axis=0)
            else:
                photons = tf.stack(photons_list, axis=1)
            loss, prob = get_loss_fn(photons, label_batch, bias=tf_bias, tanh=cfg.training.tanh, loss_type=cfg.training.loss_fn)
            
            probs.extend(prob.numpy())
            
            if (end) >= total_jets or (i+1) % 5 == 0:
                    print(f"  Processed {min(end, total_jets)}/{total_jets} jets", flush=True)

        return np.asarray(probs)

    # test ----------------------------------------------------------------
    print("Predicting on test set...", flush=True)
    prob_test = predict_prob(jets_test, labels_test)
    auc_test  = roc_auc_score(labels_test.numpy(), prob_test)

    # summary -------------------------------------------------
    print("Training completed.", flush=True)
    print(f"Final test AUC: {auc_test:.4f}", flush=True)

    # Generate and save plots
    if not cfg.runtime.cli_test:
        plots_dir = os.path.join(cfg.data.save_dir, run_name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
        score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
        plot_roc_curve(labels_test.numpy(), prob_test, roc_plot_path)
        plot_score_histogram(labels_test.numpy(), prob_test, score_hist_path)
        print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()