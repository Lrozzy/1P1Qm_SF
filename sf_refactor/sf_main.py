import strawberryfields as sf
import tensorflow as tf
import numpy as np
import os, random
import argparse
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

# ----------  hyper-params ----------
dim_cutoff      = 8 # fock cutoff dim 
wires           = 4 # number of particles per jet (1 wire per particle) DO NOT GO ABOVE 4 (memory blows up)
layers          = 1 # not using right now, but can be used for circuit depth
epochs          = 10
learning_rate   = 0.005
batch_size      = 8 # 
train_jets      = 1000 # /40000
val_jets        = 200 # /10000
test_jets       = 1000 # /20000

# Loss function and activation parameters
loss_fn         = "bce" # loss function: "bce" or "mse"
tanh            = False # True or False. Use tanh activation instead of sigmoid

# Circuit type
which_circuit   = "new"  # "default" or "new"

# Paths to data files
data_dir        = "/rds/general/user/lr1424/home/1P1Qm_SF/flat_train/TTBar+ZJets_flat.h5"
val_dir         = "/rds/general/user/lr1424/home/1P1Qm_SF/flat_val/TTBar+ZJets_flat.h5"
test_dir        = "/rds/general/user/lr1424/home/1P1Qm_SF/flat_test/TTBar+ZJets_flat.h5"
save_dir        = "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/saved_models_sf"

# Debugging
cli_test        = False # Just run the code without saving models or plots, useful for quick tests

parser = argparse.ArgumentParser(description="Run Strawberry Fields SF Simple Model")
parser.add_argument('-name', type=str, help='Name for this run (used for saving models/plots)')
parser.add_argument('--dim_cutoff', type=int, default=dim_cutoff, help='Fock cutoff dimension')
parser.add_argument('--wires', type=int, default=wires, help='Number of wires')
parser.add_argument('--layers', type=int, default=layers, help='Number of layers')
parser.add_argument('--epochs', type=int, default=epochs, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Training batch size')
parser.add_argument('--loss_fn', type=str, default=loss_fn, help='Loss function: "bce" or "mse"')
parser.add_argument('--tanh', action='store_true', help='Use tanh activation instead of sigmoid')
parser.add_argument('--which_circuit', type=str, default=which_circuit, help='Circuit type: "default" or "new"')
parser.add_argument('--train_jets', type=int, default=train_jets, help='Number of training jets')
parser.add_argument('--val_jets', type=int, default=val_jets, help='Number of validation jets')
parser.add_argument('--test_jets', type=int, default=test_jets, help='Number of test jets')
parser.add_argument('--data_dir', type=str, default=data_dir, help='Training data file')
parser.add_argument('--val_dir', type=str, default=val_dir, help='Validation data file')
parser.add_argument('--test_dir', type=str, default=test_dir, help='Test data file')
parser.add_argument('--save_dir', type=str, default=save_dir, help='Directory to save models/plots')
parser.add_argument('--cli_test', action='store_true', help='Run in CLI test mode (no saving, no plotting)')
args = parser.parse_args()

# Override hyper-params if specified in args
dim_cutoff    = args.dim_cutoff if args.dim_cutoff else dim_cutoff
wires         = args.wires if args.wires else wires
layers        = args.layers if args.layers else layers
epochs        = args.epochs if args.epochs else epochs
learning_rate = args.learning_rate if args.learning_rate else learning_rate
batch_size    = args.batch_size if args.batch_size else batch_size
loss_fn       = args.loss_fn if args.loss_fn else loss_fn
tanh          = args.tanh if args.tanh else tanh
which_circuit = args.which_circuit if args.which_circuit else which_circuit
train_jets    = args.train_jets if args.train_jets else train_jets
val_jets      = args.val_jets if args.val_jets else val_jets
test_jets     = args.test_jets if args.test_jets else test_jets
data_dir      = args.data_dir if args.data_dir else data_dir
val_dir       = args.val_dir if args.val_dir else val_dir
test_dir      = args.test_dir if args.test_dir else test_dir
save_dir      = args.save_dir if args.save_dir else save_dir
cli_test      = args.cli_test if args.cli_test else cli_test

# Adjust val_jets and test_jets to be divisible by batch_size
original_val_jets = val_jets
val_jets = ((val_jets + batch_size - 1) // batch_size) * batch_size
if original_val_jets != val_jets:
    print(f"Adjusting val_jets from {original_val_jets} to {val_jets} to be divisible by batch_size {batch_size}", flush=True)

original_test_jets = test_jets
test_jets = ((test_jets + batch_size - 1) // batch_size) * batch_size
if original_test_jets != test_jets:
    print(f"Adjusting test_jets from {original_test_jets} to {test_jets} to be divisible by batch_size {batch_size}", flush=True)


# Run name
now_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
if args.name:
    # If the provided run_name already exists, append _i to make it unique
    base_run_name = f"{now_str}_{args.name}"
    run_name = base_run_name
    i = 1
    while os.path.exists(f'{save_dir}/{run_name}'):
        run_name = f"{base_run_name}_{i}"
        i += 1
else:
    # Use just the datetime as the base name
    base_run_name = now_str
    run_name = base_run_name
    i = 1
    while os.path.exists(f'{save_dir}/{run_name}'):
        run_name = f"{base_run_name}_{i}"
        i += 1

# Get current date and time
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save parameters to a file
if cli_test == False:
    os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
    params = {
        "run_start_time": now,
        "seed": seed,
        "dim_cutoff": dim_cutoff,
        "wires": wires,
        "layers": layers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "loss_fn": loss_fn,
        "tanh": tanh,
        "which_circuit": which_circuit,
        "train_jets": train_jets,
        "val_jets": val_jets,
        "test_jets": test_jets,
        "data_dir": data_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "save_dir": save_dir,
        "run_name": run_name,
    }
    params_path = os.path.join(save_dir, run_name, "params.txt")
    with open(params_path, "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
            
# -----------------------------------
# Print parameters
print(f"Run started at: {now}", flush=True)
print("PARAMETERS:")
print(f"Random seed for this run: {seed}", flush=True)
print(f"Dimension cutoff: {dim_cutoff}", flush=True)
print(f"Wires: {wires}", flush=True)
print(f"Layers: {layers}", flush=True)
print(f"Epochs: {epochs}", flush=True)
print(f"Learning rate: {learning_rate}", flush=True)
print(f"Batch size: {batch_size}", flush=True)
print(f"Loss function: {loss_fn}", flush=True)
if tanh:
    print("\t Tanh activation: True", flush=True)
else:
    print("\t Sigmoid activation with trainable bias", flush=True)
print(f"Which circuit: {which_circuit}", flush=True)
print(f"Train jets: {train_jets}", flush=True)
print(f"Validation jets: {val_jets}", flush=True)
print(f"Test jets: {test_jets}", flush=True)
print(f"Run name: {run_name}", flush=True)
print("------------------------------------", flush=True)

# ----------  load datasets ---------- 
jets, labels = load_data(data_dir, max_jets=train_jets, wires=wires)
jets_val, labels_val = load_data(val_dir, max_jets=val_jets, wires=wires)
jets_test, labels_test = load_data(test_dir, max_jets=test_jets, wires=wires)

# -------- symbolic circuit ----------
# print("Starting symbolic circuit construction...", flush=True)
prog = sf.Program(wires)
s_scale = prog.params("s_scale")
disp_mag  = [prog.params(f"disp_mag{w}") for w in range(wires)] # disp_mag = Displacement Magnitude
disp_phase  = [prog.params(f"disp_phase{w}") for w in range(wires)] # disp_phase = Displacement Phase
squeeze_mag  = [prog.params(f"squeeze_mag{w}") for w in range(wires)] # squeeze_mag = Squeezing Magnitude
squeeze_phase  = [prog.params(f"squeeze_phase{w}") for w in range(wires)] # squeeze_phase = Squeezing Phase
eta = [prog.params(f"eta{w}") for w in range(wires)]
phi = [prog.params(f"phi{w}") for w in range(wires)]
pt  = [prog.params(f"pt{w}")  for w in range(wires)]

# Extra variables for trainable cx gates
cx_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
cx_theta = { (a,b): prog.params(f"cx_theta_{a}_{b}") for (a,b) in cx_pairs }

# -------- Circuit architecture ----------
if which_circuit == "new":
    weights = {
        's_scale': s_scale,
        **{f'disp_mag_{w}': disp_mag[w] for w in range(wires)},
        **{f'disp_phase_{w}': disp_phase[w] for w in range(wires)},
        **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(wires)},
        **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(wires)},
        **{f'eta_{w}': eta[w] for w in range(wires)},
        **{f'phi_{w}': phi[w] for w in range(wires)},
        **{f'pt_{w}': pt[w] for w in range(wires)},
        **{f"cx_theta_{a}_{b}": cx_theta[(a,b)] for (a,b) in cx_pairs},
    }
    prog = new_circuit(prog, wires, weights)
else:
    weights = {
        's_scale': s_scale,
        **{f'disp_mag_{w}': disp_mag[w] for w in range(wires)},
        **{f'disp_phase_{w}': disp_phase[w] for w in range(wires)},
        **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(wires)},
        **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(wires)},
        **{f'eta_{w}': eta[w] for w in range(wires)},
        **{f'phi_{w}': phi[w] for w in range(wires)},
        **{f'pt_{w}': pt[w] for w in range(wires)},
    }
    prog = default_circuit(prog, wires, weights)

# -------- Initialise variables ----------
# print("Initialising variables...", flush=True)
rnd = tf.random_uniform_initializer(-0.1, 0.1)
tf_s_scale = tf.Variable(rnd(()))
tf_disp_mag = [tf.Variable(rnd(())) for _ in range(wires)]
tf_disp_phase = [tf.Variable(rnd(())) for _ in range(wires)]
tf_squeeze_mag = [tf.Variable(rnd(())) for _ in range(wires)]
tf_squeeze_phase = [tf.Variable(rnd(())) for _ in range(wires)]
tf_bias = tf.Variable(0.0, dtype=tf.float32) # Trainable bias

# Extra variables for trainable cx gates
tf_cx_theta = { (a,b): tf.Variable(rnd(())) for (a,b) in cx_pairs }

# -------- Feature scaling ----------
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
    d = {"s_scale": tf_s_scale}
    for w in range(wires):
        d[f"disp_mag{w}"] = tf_disp_mag[w]
        d[f"disp_phase{w}"] = tf_disp_phase[w]
        d[f"squeeze_mag{w}"] = tf_squeeze_mag[w]
        d[f"squeeze_phase{w}"] = tf_squeeze_phase[w]
        # Slicing the batch to get all values for a specific feature across the batch
        d[f"eta{w}"] = scale_feature(jet_batch[:, w, 0], "eta")
        d[f"phi{w}"] = scale_feature(jet_batch[:, w, 1], "phi")
        d[f"pt{w}"]  = scale_feature(jet_batch[:, w, 2], "pt")
    if which_circuit == "new":
        for (a, b) in cx_pairs:
            d[f"cx_theta_{a}_{b}"] = tf_cx_theta[(a, b)]
    return d

opt = tf.keras.optimizers.Adam(learning_rate)
# print("Starting Engine...", flush=True)
eng = sf.Engine("tf", backend_options={"cutoff_dim": dim_cutoff, "batch_size": batch_size})

# -------- training loop ----------
print("Starting training...", flush=True)

# Create a tf.data.Dataset for efficient batching and shuffling
train_dataset = tf.data.Dataset.from_tensor_slices((jets, labels))
train_dataset = train_dataset.shuffle(buffer_size=train_jets).batch(batch_size, drop_remainder=True)

for epoch in range(epochs):
    # Keep track of the loss for this epoch
    epoch_loss = 0.0
    num_steps = 0
    # Use a dictionary to store gradient norms for each variable
    epoch_grad_norms = {}
    
    # Loop over batches in the training set
    if cli_test:
        train_iterator = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch+1}/{epochs}")
    else:
        train_iterator = enumerate(train_dataset)

    for step, (jet_batch, label_batch) in train_iterator:
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            # Run the circuit for the entire batch
            state   = eng.run(prog, args=make_args(jet_batch)).state
            # state.mean_photon(m) returns a tuple (mean, variance), we only want the mean
            photons = tf.stack([state.mean_photon(m)[0] for m in range(3)], axis=1)
            # Calculate loss for the batch
            loss_vector, logit_to_prob = get_loss_fn(photons, label_batch, bias=tf_bias, tanh=tanh, loss_type=loss_fn)
            # Average the loss over the batch for a stable gradient
            loss = tf.reduce_mean(loss_vector)

        if which_circuit == "new":
            var_names = ['s_scale'] + [f'disp_mag_{i}' for i in range(wires)] +                 [f'disp_phase_{i}' for i in range(wires)] +                 [f'squeeze_mag_{i}' for i in range(wires)] +                 [f'squeeze_phase_{i}' for i in range(wires)] +                 [f'cx_theta_{a}_{b}' for a,b in cx_pairs] + ['bias']
            vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, *tf_cx_theta.values(), tf_bias]
        else:
            var_names = ['s_scale'] + [f'disp_mag_{i}' for i in range(wires)] +                 [f'disp_phase_{i}' for i in range(wires)] +                 [f'squeeze_mag_{i}' for i in range(wires)] +                 [f'squeeze_phase_{i}' for i in range(wires)] + ['bias']
            vars_ = [tf_s_scale, *tf_disp_mag, *tf_disp_phase, *tf_squeeze_mag, *tf_squeeze_phase, tf_bias]
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        
        epoch_loss += loss.numpy()
        num_steps += 1

        if cli_test:
            # Store the norm of each gradient for each variable
            for i, g in enumerate(grads):
                if g is not None:
                    var_name = var_names[i]
                    if var_name not in epoch_grad_norms:
                        epoch_grad_norms[var_name] = []
                    epoch_grad_norms[var_name].append(tf.norm(g).numpy())
            train_iterator.set_postfix(loss=f"{loss.numpy():.4f}")

        if not cli_test and (step + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Step {step+1}/{len(train_dataset)} - Batch Loss: {loss:.4f}", flush=True)

    # -------- validation step at the end of each epoch ----------
    avg_train_loss = epoch_loss / num_steps
    val_probs_pass = []
    val_losses_pass = []
    
    # Process validation set in batches
    num_val_batches = val_jets // batch_size
    for i in range(num_val_batches):
        start = i * batch_size
        end = start + batch_size
        jet_batch_val = jets_val[start:end]
        label_batch_val = labels_val[start:end]

        if eng.run_progs:
            eng.reset()

        state   = eng.run(prog, args=make_args(jet_batch_val)).state
        photons = tf.stack([state.mean_photon(m)[0] for m in range(3)], axis=1)
        val_loss_vector, val_prob = get_loss_fn(photons, label_batch_val, bias=tf_bias, tanh=tanh, loss_type=loss_fn)
        
        val_probs_pass.extend(val_prob.numpy())
        val_losses_pass.extend(val_loss_vector.numpy())

    avg_val_loss = np.mean(val_losses_pass)
    auc_val = roc_auc_score(labels_val.numpy(), np.asarray(val_probs_pass))
    
    if cli_test:
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation AUC: {auc_val:.4f}", flush=True)
        # Diagnostic: Print the first 5 validation probabilities to check if they are changing
        val_probs_preview = ", ".join([f"{p:.4f}" for p in val_probs_pass[:5]])
        print(f"  Validation Probs Preview: [{val_probs_preview}]", flush=True)
        print("  Average Gradient Norms per variable for this epoch:", flush=True)
        for var_name, norms in sorted(epoch_grad_norms.items()):
            avg_norm = np.mean(norms) if norms else 0.0
            print(f"    {var_name}: {avg_norm:.6f}", flush=True)
    else:
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation AUC: {auc_val:.4f}", flush=True)

# --------- Evaluate and print AUC ---------
def predict_prob(jets_tensor, labels):
    """Return an array of P(signal) for each jet."""
    probs = []
    total_jets = jets_tensor.shape[0]
    num_batches = total_jets // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        jet_batch = jets_tensor[start:end]
        label_batch = labels[start:end]

        if eng.run_progs:
            eng.reset()
        state   = eng.run(prog, args=make_args(jet_batch)).state
        photons = tf.stack([state.mean_photon(m)[0] for m in range(3)], axis=1)
        loss, prob = get_loss_fn(photons, label_batch, bias=tf_bias, tanh=tanh, loss_type=loss_fn)
        
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
if cli_test == False:
    plots_dir = os.path.join(save_dir, run_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    roc_plot_path = os.path.join(plots_dir, 'roc_curve.png')
    score_hist_path = os.path.join(plots_dir, 'score_histogram.png')
    plot_roc_curve(labels_test.numpy(), prob_test, roc_plot_path)
    plot_score_histogram(labels_test.numpy(), prob_test, score_hist_path)
    print(f"Plots saved to {plots_dir}")