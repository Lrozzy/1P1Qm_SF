"""
Model saving and loading utilities for StrawberryFields quantum circuits.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
import strawberryfields as sf
from circuits import default_circuit, new_circuit


def save_quantum_model(model_weights, config, run_name, save_dir):
    """
    Save a trained quantum model with all necessary components.
    
    Args:
        model_weights: Dictionary containing all trained parameter values
        config: Configuration object used for training
        run_name: Name of the training run
        save_dir: Base directory for saving models
    """
    # Create model directory
    model_dir = os.path.join(save_dir, run_name, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    weights_path = os.path.join(model_dir, "model_weights.pkl")
    with open(weights_path, 'wb') as f:
        pickle.dump(model_weights, f)
    
    # Save model configuration
    config_path = os.path.join(model_dir, "model_config.yaml")
    OmegaConf.save(config, config_path)
    
    # Save feature scaling parameters (from training script)
    feature_scaling = {
        'assumed_limits': {
            'pt': [1e-4, 3000.0],
            'eta': [-0.8, 0.8],
            'phi': [-0.8, 0.8],
        },
        'feature_limits': {
            'pt': [0.0, 1.0],
            'eta': [-np.pi, np.pi],
            'phi': [-np.pi, np.pi],
        }
    }
    
    scaling_path = os.path.join(model_dir, "feature_scaling.pkl")
    with open(scaling_path, 'wb') as f:
        pickle.dump(feature_scaling, f)
    
    # Save circuit architecture information
    circuit_info = {
        'circuit_type': config.model.which_circuit,
        'circuit_file': 'circuits.py',
        'circuit_function': f"{config.model.which_circuit}_circuit",
        'wires': config.model.wires,
        'photon_modes': config.model.photon_modes,
        'dim_cutoff': config.model.dim_cutoff,
        'cx_pairs': [(i, j) for i in range(config.model.wires) for j in range(i+1, config.model.wires)] if config.model.which_circuit == "new" else None
    }
    
    circuit_path = os.path.join(model_dir, "circuit_architecture.pkl")
    with open(circuit_path, 'wb') as f:
        pickle.dump(circuit_info, f)
    
    # Save metadata
    metadata = {
        'model_type': 'strawberryfields_quantum',
        'framework_version': sf.__version__,
        'tensorflow_version': tf.__version__,
        'save_timestamp': np.datetime64('now').item().isoformat(),
        'circuit_architecture': config.model.which_circuit,
        'feature_scaling_saved': True
    }
    
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model saved to: {model_dir}")
    print(f"  - Model weights: model_weights.pkl")
    print(f"  - Configuration: model_config.yaml") 
    print(f"  - Feature scaling: feature_scaling.pkl")
    print(f"  - Circuit architecture: circuit_architecture.pkl")
    print(f"  - Metadata: metadata.pkl")
    
    return model_dir


def load_quantum_model(model_dir):
    """
    Load a saved quantum model.
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        tuple: (model_weights, config, metadata, feature_scaling, circuit_info)
    """
    # Load weights
    weights_path = os.path.join(model_dir, "model_weights.pkl")
    with open(weights_path, 'rb') as f:
        model_weights = pickle.load(f)
    
    # Load configuration
    config_path = os.path.join(model_dir, "model_config.yaml")
    config = OmegaConf.load(config_path)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load feature scaling parameters
    scaling_path = os.path.join(model_dir, "feature_scaling.pkl")
    if os.path.exists(scaling_path):
        with open(scaling_path, 'rb') as f:
            feature_scaling = pickle.load(f)
    else:
        # Fallback to hardcoded values for backwards compatibility
        feature_scaling = {
            'assumed_limits': {
                'pt': [1e-4, 3000.0],
                'eta': [-0.8, 0.8],
                'phi': [-0.8, 0.8],
            },
            'feature_limits': {
                'pt': [0.0, 1.0],
                'eta': [-np.pi, np.pi],
                'phi': [-np.pi, np.pi],
            }
        }
        print("Warning: Using hardcoded feature scaling (older model format)")
    
    # Load circuit architecture information
    circuit_path = os.path.join(model_dir, "circuit_architecture.pkl")
    if os.path.exists(circuit_path):
        with open(circuit_path, 'rb') as f:
            circuit_info = pickle.load(f)
    else:
        # Fallback for backwards compatibility
        circuit_info = {
            'circuit_type': config.model.which_circuit,
            'circuit_file': 'circuits.py',
            'circuit_function': f"{config.model.which_circuit}_circuit",
            'wires': config.model.wires,
            'photon_modes': config.model.photon_modes,
            'dim_cutoff': config.model.dim_cutoff,
            'cx_pairs': [(i, j) for i in range(config.model.wires) for j in range(i+1, config.model.wires)] if config.model.which_circuit == "new" else None
        }
        print("Warning: Using inferred circuit architecture (older model format)")
    
    return model_weights, config, metadata, feature_scaling, circuit_info

