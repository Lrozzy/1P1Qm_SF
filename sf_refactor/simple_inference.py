#!/usr/bin/env python3
"""
Simple inference script for trained quantum models.
Usage: python simple_inference.py --model_dir path/to/saved/model [--max_jets 1000]
"""

import os
import argparse
import strawberryfields as sf
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from helpers.model_utils import load_quantum_model
from helpers.utils import load_data, get_loss_fn
from circuits import default_circuit, new_circuit

def run_inference(model_dir, max_jets=None, batch_size=32, cli_mode=False, random_subset=False, seed=None):
    """
    Run inference using a saved quantum model.
    
    Args:
        model_dir: Path to saved model directory
        max_jets: Maximum number of jets to process (None = use config default)
        batch_size: Batch size for inference
        cli_mode: If True, use tqdm progress bar; if False, use simple print statements
        random_subset: If True, randomly sample from test data; if False, use first N jets
        seed: Random seed for reproducible random sampling (only used if random_subset=True)
    """
    
    # Load the saved model
    print("Loading saved model...")
    model_weights, config, metadata, feature_scaling, circuit_info = load_quantum_model(model_dir)
    print(f"Loaded model: {metadata['circuit_architecture']} circuit")
    
    # Format timestamp for display (convert 2025-07-17T22:40:34 to 2025-07-17 22:40:34)
    formatted_timestamp = metadata['save_timestamp'].replace('T', ' ')
    print(f"Model saved: {formatted_timestamp}")
    
    # Use config test data settings if max_jets not specified
    if max_jets is None:
        max_jets = config.data.test_jets
    
    # Load test data using config paths
    print(f"Loading test data from: {config.data.test_dir}")
    print(f"Processing {max_jets} jets...")
    jets_test, labels_test = load_data(config.data.test_dir, max_jets=config.data.test_jets, wires=config.model.wires)
    
    # Apply random subset sampling if requested
    if random_subset:
        if seed is not None:
            np.random.seed(seed)
            print(f"Using random seed: {seed}")
        
        total_available = jets_test.shape[0]
        if max_jets < total_available:
            print(f"Randomly sampling {max_jets} jets from {total_available} available jets")
            indices = np.random.choice(total_available, size=max_jets, replace=False)
            jets_test = jets_test[indices]
            labels_test = labels_test[indices]
        else:
            print(f"Requested {max_jets} jets, but only {total_available} available - using all")
    else:
        # Use first max_jets (deterministic)
        if max_jets < jets_test.shape[0]:
            print(f"Using first {max_jets} jets (deterministic)")
            jets_test = jets_test[:max_jets]
            labels_test = labels_test[:max_jets]
    
    # Create symbolic circuit (same as training)
    print("Setting up quantum circuit...")
    prog = sf.Program(config.model.wires)
    
    # Create symbolic parameters
    s_scale = prog.params("s_scale")
    disp_mag = [prog.params(f"disp_mag{w}") for w in range(config.model.wires)]
    disp_phase = [prog.params(f"disp_phase{w}") for w in range(config.model.wires)]
    squeeze_mag = [prog.params(f"squeeze_mag{w}") for w in range(config.model.wires)]
    squeeze_phase = [prog.params(f"squeeze_phase{w}") for w in range(config.model.wires)]
    eta = [prog.params(f"eta{w}") for w in range(config.model.wires)]
    phi = [prog.params(f"phi{w}") for w in range(config.model.wires)]
    pt = [prog.params(f"pt{w}") for w in range(config.model.wires)]
    
    # Extra parameters for new circuit
    cx_pairs = [(i, j) for i in range(config.model.wires) for j in range(i+1, config.model.wires)]
    cx_theta = {(a,b): prog.params(f"cx_theta_{a}_{b}") for (a,b) in cx_pairs}
    
    # Build circuit
    if config.model.which_circuit == "new":
        weights = {
            's_scale': s_scale,
            **{f'disp_mag_{w}': disp_mag[w] for w in range(config.model.wires)},
            **{f'disp_phase_{w}': disp_phase[w] for w in range(config.model.wires)},
            **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(config.model.wires)},
            **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(config.model.wires)},
            **{f'eta_{w}': eta[w] for w in range(config.model.wires)},
            **{f'phi_{w}': phi[w] for w in range(config.model.wires)},
            **{f'pt_{w}': pt[w] for w in range(config.model.wires)},
            **{f"cx_theta_{a}_{b}": cx_theta[(a,b)] for (a,b) in cx_pairs},
        }
        prog = new_circuit(prog, config.model.wires, weights)
    else:
        weights = {
            's_scale': s_scale,
            **{f'disp_mag_{w}': disp_mag[w] for w in range(config.model.wires)},
            **{f'disp_phase_{w}': disp_phase[w] for w in range(config.model.wires)},
            **{f'squeeze_mag_{w}': squeeze_mag[w] for w in range(config.model.wires)},
            **{f'squeeze_phase_{w}': squeeze_phase[w] for w in range(config.model.wires)},
            **{f'eta_{w}': eta[w] for w in range(config.model.wires)},
            **{f'phi_{w}': phi[w] for w in range(config.model.wires)},
            **{f'pt_{w}': pt[w] for w in range(config.model.wires)},
        }
        prog = default_circuit(prog, config.model.wires, weights)
    
    # Feature scaling function (from saved parameters)
    def scale_feature(value, name):
        a_min, a_max = feature_scaling['assumed_limits'][name]
        f_min, f_max = feature_scaling['feature_limits'][name]
        return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min
    
    # Function to create arguments for SF program (using saved weights)
    def make_args(jet_batch):
        squeeze = jet_batch.shape[0] == 1
        
        d = {"s_scale": model_weights["s_scale"]}
        
        for w in range(config.model.wires):
            d[f"disp_mag{w}"] = model_weights[f"disp_mag_{w}"]
            d[f"disp_phase{w}"] = model_weights[f"disp_phase_{w}"]
            d[f"squeeze_mag{w}"] = model_weights[f"squeeze_mag_{w}"]
            d[f"squeeze_phase{w}"] = model_weights[f"squeeze_phase_{w}"]
            
            # Scale features (same as training)
            eta_val = scale_feature(jet_batch[:, w, 0], "eta")
            phi_val = scale_feature(jet_batch[:, w, 1], "phi")
            pt_val = scale_feature(jet_batch[:, w, 2], "pt")
            
            if squeeze:
                d[f"eta{w}"] = tf.squeeze(eta_val)
                d[f"phi{w}"] = tf.squeeze(phi_val)
                d[f"pt{w}"] = tf.squeeze(pt_val)
            else:
                d[f"eta{w}"] = eta_val
                d[f"phi{w}"] = phi_val
                d[f"pt{w}"] = pt_val
        
        # Add cx parameters if new circuit
        if config.model.which_circuit == "new":
            for (a, b) in cx_pairs:
                d[f"cx_theta_{a}_{b}"] = model_weights[f"cx_theta_{a}_{b}"]
        
        return d
    
    # Initialize engine
    sf_batch_size = None if batch_size == 1 else batch_size
    eng = sf.Engine("tf", backend_options={"cutoff_dim": config.model.dim_cutoff, "batch_size": sf_batch_size})
    
    # Run inference
    print("Running inference...")
    probs = []
    bias = model_weights["bias"]
    total_jets = jets_test.shape[0]
    num_batches = total_jets // batch_size
    processed_jets = 0
    
    # Setup progress display
    if cli_mode:
        batch_iterator = tqdm(range(num_batches), desc="Processing batches", unit="batch")
    else:
        batch_iterator = range(num_batches)
    
    for i in batch_iterator:
        start = i * batch_size
        end = start + batch_size
        jet_batch = jets_test[start:end]
        label_batch = labels_test[start:end]
        
        if eng.run_progs:
            eng.reset()
            
        state = eng.run(prog, args=make_args(jet_batch)).state
        photons_list = [state.mean_photon(m)[0] for m in range(config.model.photon_modes)]
        
        if batch_size == 1:
            photons = tf.expand_dims(tf.stack(photons_list, axis=0), axis=0)
        else:
            photons = tf.stack(photons_list, axis=1)
            
        _, prob = get_loss_fn(photons, label_batch, bias=bias, tanh=config.training.tanh, loss_type=config.training.loss_fn)
        probs.extend(prob.numpy())
        processed_jets += jet_batch.shape[0]
        
        # Update progress for non-CLI mode
        if not cli_mode and ((i+1) % 10 == 0 or (i+1) == num_batches):
            print(f"  Processed {processed_jets}/{total_jets} jets")
    
    # Calculate final AUC
    probs = np.array(probs)
    true_labels = labels_test[:len(probs)].numpy()
    auc = roc_auc_score(true_labels, probs)
    
    print(f"\nInference completed!")
    print(f"Final Test AUC: {auc:.4f}")
    print(f"Total jets processed: {len(probs)}")
    
    return probs, true_labels, auc


def main():
    parser = argparse.ArgumentParser(description='Run inference on saved quantum model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory')
    parser.add_argument('--max_jets', type=int, default=None,
                        help='Maximum number of jets to process (default: use config setting)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--cli_test', action='store_true',
                        help='Enable CLI mode with tqdm progress bar (default: False for background jobs)')
    parser.add_argument('--random_subset', action='store_true',
                        help='Randomly sample jets from test data instead of using first N jets')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible random sampling (only used with --random_subset)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist")
        return
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        probs, labels, auc = run_inference(args.model_dir, args.max_jets, args.batch_size, 
                                         args.cli_test, args.random_subset, args.seed)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
