#!/usr/bin/env python3
"""
X8 Hardware inference script for trained quantum models.
Submits jobs to Xanadu X8 quantum photonic hardware.
"""

import strawberryfields as sf
from strawberryfields import RemoteEngine
import numpy as np
import argparse
import os
import time
import subprocess
import json
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Xanadu Cloud Connection
import xcc
import xcc.commands

from helpers.model_utils import load_quantum_model
from helpers.utils import load_data, get_loss_fn
from circuits import x8_circuit_hardware

def check_xanadu_connection():
    """Verify connection to Xanadu Cloud."""
    print("Checking Xanadu Cloud connection...", flush=True)
    try:
        connection = xcc.Connection.load()
        ping_result = connection.ping()
        assert ping_result.ok, f"Connection failed: {ping_result}"
        
        # Additional ping test
        xcc.commands.ping()
        print("Successfully connected to Xanadu Cloud", flush=True)
        return True
    except Exception as e:
        print(f"Failed to connect to Xanadu Cloud: {e}", flush=True)
        return False

def check_x8_device_availability():
    """Check if X8_01 device is currently online using xcc command."""
    try:
        # Run xcc device list --status online
        result = subprocess.run(
            ['xcc', 'device', 'list', '--status', 'online'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse JSON output
        devices = json.loads(result.stdout)
        
        # Look for X8_01 device specifically
        x8_devices = [device for device in devices if device.get("target") == "X8_01"]
        
        if x8_devices:
            x8_device = x8_devices[0]
            print(f"Found X8_01 device: {x8_device['status']}", flush=True)
            if x8_device['status'] == 'online':
                return True, x8_device['target']
            else:
                print(f"X8_01 is {x8_device['status']}, not available", flush=True)
                return False, None
        else:
            # Check if X8_01 exists but is offline by checking ALL devices
            result_all = subprocess.run(
                ['xcc', 'device', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            all_devices = json.loads(result_all.stdout)
            x8_all = [device for device in all_devices if device.get("target") == "X8_01"]
            
            if x8_all:
                x8_device = x8_all[0]
                print(f"X8_01 found but offline: {x8_device['status']}", flush=True)
            else:
                print("X8_01 device not found in device list", flush=True)
            
            print("Available online devices:", flush=True)
            for device in devices:
                print(f"  - {device['target']}: {device['status']}", flush=True)
            return False, None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running xcc device list: {e}", flush=True)
        print(f"stderr: {e.stderr}", flush=True)
        return False, None
    except json.JSONDecodeError as e:
        print(f"Error parsing device list JSON: {e}", flush=True)
        return False, None
    except Exception as e:
        print(f"Unexpected error checking device availability: {e}", flush=True)
        return False, None

def wait_for_x8_availability(max_wait_time=3600*12, check_interval=300):
    """
    Wait for X8 devices to become available.
    
    Args:
        max_wait_time: Maximum time to wait in seconds (default: 12 hours)
        check_interval: Time between checks in seconds (default: 5 minutes)

    Returns:
        (bool, str): (True, device_target) if device becomes available, (False, None) if timeout
    """
    print(f"Waiting for X8 devices to come online (max wait: {max_wait_time//60} minutes)...", flush=True)
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        x8_available, x8_target = check_x8_device_availability()
        if x8_available:
            return True, x8_target
        
        remaining_time = max_wait_time - (time.time() - start_time)
        print(f"No X8 devices online. Checking again in {check_interval}s (remaining: {remaining_time//60:.1f}m)...", flush=True)
        time.sleep(check_interval)
    
    print(f"Timeout: No X8 devices came online within {max_wait_time//60} minutes", flush=True)
    return False, None
    
def scale_feature(value, name, feature_scaling):
    """Feature scaling function (imported from simple_inference.py logic)."""
    a_min, a_max = feature_scaling['assumed_limits'][name]
    f_min, f_max = feature_scaling['feature_limits'][name]
    return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

def get_concrete_values(jet_batch, config, feature_scaling):
    """Extract concrete values from jet data for X8 hardware circuit construction.
    
    Args:
        jet_batch: Single jet data array
        config: Model configuration
        feature_scaling: Feature scaling parameters
    
    Returns:
        Dictionary of concrete values for circuit construction
    """
    squeeze = jet_batch.shape[0] == 1
    
    # Extract and scale jet features
    concrete_values = {}
    
    for w in range(config.model.wires):
        # Scale features (same as training)
        eta_val = scale_feature(jet_batch[:, w, 0], "eta", feature_scaling)
        phi_val = scale_feature(jet_batch[:, w, 1], "phi", feature_scaling)
        pt_val = scale_feature(jet_batch[:, w, 2], "pt", feature_scaling)
        
        if squeeze:
            concrete_values[f'eta_{w}'] = float(tf.squeeze(eta_val))
            concrete_values[f'phi_{w}'] = float(tf.squeeze(phi_val))
            concrete_values[f'pt_{w}'] = float(tf.squeeze(pt_val))
        else:
            concrete_values[f'eta_{w}'] = float(eta_val)
            concrete_values[f'phi_{w}'] = float(phi_val)
            concrete_values[f'pt_{w}'] = float(pt_val)
    
    return concrete_values

def run_x8_hardware_inference(model_dir, max_jets=100, batch_size=1, shots=10000, wait_for_device=True):
    """
    Run inference on X8 hardware.
    
    Args:
        model_dir: Path to saved model
        max_jets: Number of jets to process
        batch_size: Batch size for inference
        shots: Number of measurement shots per circuit execution
        wait_for_device: Whether to wait for X8 devices to come online
    """
    
    # Check cloud connection
    if not check_xanadu_connection():
        return None, None, None
    
    # Load saved model (reuse from simple_inference.py)
    print("Loading saved model...", flush=True)
    model_weights, config, metadata, feature_scaling, circuit_info = load_quantum_model(model_dir)
    
    print(f"Loaded model: {config.model.which_circuit} circuit", flush=True)
    print(f"Model saved: {metadata['save_timestamp'].replace('T', ' ')}", flush=True)
    print(f"Running inference with: X8 hardware circuit", flush=True)
    
    # Load test data (reuse from simple_inference.py)
    print(f"Loading test data from: {config.data.test_dir}", flush=True)
    jets_test, labels_test = load_data(config.data.test_dir, max_jets=max_jets, wires=config.model.wires)
    
    if max_jets and max_jets < len(jets_test):
        jets_test = jets_test[:max_jets]
        labels_test = labels_test[:max_jets]
    
    print(f"Processing {len(jets_test)} jets with {shots} shots per execution...", flush=True)
    
    # Define CX pairs for coupling parameters
    cx_pairs = [(i, j) for i in range(config.model.wires) for j in range(i+1, config.model.wires)]
    
    # Check if X8 devices are available before proceeding
    print("Checking X8 device availability...", flush=True)
    x8_available, x8_target = check_x8_device_availability()
    
    if not x8_available:
        if wait_for_device:
            print("X8_01 is not online. Waiting for it to become available...", flush=True)
            x8_available, x8_target = wait_for_x8_availability()
            if not x8_available:
                print("Timeout waiting for X8_01. Exiting...", flush=True)
                return None, None, None
        else:
            print("X8_01 is not online. Exiting...", flush=True)
            return None, None, None
    
    print(f"Using X8 device: {x8_target}", flush=True)
    
    eng = RemoteEngine("X8")  # Real X8 hardware
    
    # Run inference with job submission
    print(f"Submitting inference jobs to X8 hardware...", flush=True)
    probs = []
    bias = model_weights["bias"]
    total_jets = jets_test.shape[0]
    num_batches = total_jets // batch_size
    processed_jets = 0
    
    # Process in batches - each jet requires rebuilding the circuit with its specific values
    for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        try:
            start = i * batch_size
            end = start + batch_size
            jet_batch = jets_test[start:end]
            label_batch = labels_test[start:end]
            
            # Get concrete values for this specific jet
            jet_concrete = get_concrete_values(jet_batch, config, feature_scaling)

            # Create named program for X8 job submission
            jet_prog = sf.Program(8, name=f"quantum_jet_inference_batch_{i}")
            
            jet_weights = {
                's_scale': float(model_weights["s_scale"]),
                # Signal modes 0-3: use this jet's concrete values
                **{f'disp_mag_{w}': float(model_weights[f'disp_mag_{w}']) for w in range(config.model.wires)},
                **{f'disp_phase_{w}': float(model_weights[f'disp_phase_{w}']) for w in range(config.model.wires)},
                **{f'squeeze_mag_{w}': float(model_weights[f'squeeze_mag_{w}']) for w in range(config.model.wires)},
                **{f'squeeze_phase_{w}': float(model_weights[f'squeeze_phase_{w}']) for w in range(config.model.wires)},
                **{f'eta_{w}': jet_concrete[f'eta_{w}'] for w in range(config.model.wires)},
                **{f'phi_{w}': jet_concrete[f'phi_{w}'] for w in range(config.model.wires)},
                **{f'pt_{w}': jet_concrete[f'pt_{w}'] for w in range(config.model.wires)},
                # Idler modes 4-7: duplicate signal mode values
                **{f'disp_mag_{w+4}': float(model_weights[f'disp_mag_{w}']) for w in range(config.model.wires)},
                **{f'disp_phase_{w+4}': float(model_weights[f'disp_phase_{w}']) for w in range(config.model.wires)},
                **{f'squeeze_mag_{w+4}': float(model_weights[f'squeeze_mag_{w}']) for w in range(config.model.wires)},
                **{f'squeeze_phase_{w+4}': float(model_weights[f'squeeze_phase_{w}']) for w in range(config.model.wires)},
                **{f'eta_{w+4}': jet_concrete[f'eta_{w}'] for w in range(config.model.wires)},
                **{f'phi_{w+4}': jet_concrete[f'phi_{w}'] for w in range(config.model.wires)},
                **{f'pt_{w+4}': jet_concrete[f'pt_{w}'] for w in range(config.model.wires)},
                # CX coupling parameters (use model weights, not sample-dependent)
                **{f"cx_theta_{a}_{b}": float(model_weights[f"cx_theta_{a}_{b}"]) for (a,b) in cx_pairs},
                **{f"cx_theta_{a+4}_{b+4}": float(model_weights[f"cx_theta_{a}_{b}"]) for (a,b) in cx_pairs},
            }
            
            jet_prog = x8_circuit_hardware(jet_prog, config.model.wires, jet_weights)
            
            # Submit job asynchronously to X8 hardware
            print(f"Submitting batch {i+1}/{num_batches} to X8...", flush=True)
            job = eng.run_async(jet_prog, shots=shots)
            print(f"Job {job.id} submitted, waiting for completion...", flush=True)
            
            # Wait for job completion
            job.wait()
            print(f"Job {job.id} completed with status: {job.status}", flush=True)
            
            # Get results from completed job
            if job.status == "complete":
                result = sf.Result(job.result)
                print(f"Successfully retrieved results from X8 hardware", flush=True)
            else:
                raise Exception(f"Job failed with status: {job.status}")
            
            # Extract mean photon numbers from samples
            # result.samples has shape (shots, 8) for 8-mode X8
            photons_mean = np.mean(result.samples, axis=0)  # Mean over shots
            
            # Use only the first 4 modes (signal modes) matching training
            photons_list = photons_mean[:config.model.photon_modes]
            
            # Convert to tensor format expected by loss function
            if batch_size == 1:
                photons = tf.expand_dims(tf.constant(photons_list, dtype=tf.float32), axis=0)
            else:
                # For batches > 1, need to handle multiple jets
                # This assumes X8 can handle batched execution (may need verification)
                photons = tf.constant(photons_list, dtype=tf.float32)
                photons = tf.expand_dims(photons, axis=0)  # Add batch dimension
            
            # Calculate probability (reuse from simple_inference.py)
            _, prob = get_loss_fn(photons, label_batch, 
                                 bias=bias, 
                                 tanh=config.training.tanh, 
                                 loss_type=config.training.loss_fn)
            probs.extend(prob.numpy())
            processed_jets += jet_batch.shape[0]
            
        except Exception as e:
            print(f"Failed to process batch {i}: {e}", flush=True)
            continue
    
    # Calculate final AUC (reuse from simple_inference.py)
    if len(probs) > 0:
        probs = np.array(probs)
        true_labels = labels_test[:len(probs)].numpy()
        auc = roc_auc_score(true_labels, probs)
        
        print(f"\nX8 Hardware Inference completed!", flush=True)
        print(f"Final Test AUC: {auc:.4f}", flush=True)
        print(f"Total jets processed: {len(probs)}", flush=True)
        
        return probs, true_labels, auc
    else:
        print("No successful results obtained!", flush=True)
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Run inference on X8 quantum hardware')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory')
    parser.add_argument('--max_jets', type=int, default=100,
                        help='Maximum number of jets to process (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--shots', type=int, default=10000,
                        help='Number of measurement shots per execution (default: 10000)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for X8 devices if none are currently available (exit immediately)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist", flush=True)
        return
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    print("=" * 60, flush=True)
    print("X8 QUANTUM HARDWARE INFERENCE", flush=True)
    print("=" * 60, flush=True)
    
    try:
        probs, labels, auc = run_x8_hardware_inference(
            args.model_dir, 
            max_jets=args.max_jets,
            batch_size=args.batch_size,
            shots=args.shots,
            wait_for_device=not args.no_wait
        )
        
        if auc is not None:
            print(f"\nSuccess! X8 Hardware AUC: {auc:.4f}", flush=True)
        else:
            print("\nInference failed!", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user", flush=True)
    except Exception as e:
        print(f"\nError during inference: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
