"""Configuration management utilities for Hydra-based SF training."""

import os
import h5py
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


def validate_and_adjust_config(cfg: DictConfig) -> DictConfig:
    """Validate and adjust configuration parameters"""
    # Ensure photon_modes doesn't exceed wires
    cfg.model.photon_modes = min(cfg.model.photon_modes, cfg.model.wires)
    
    # Calculate and store particles needed based on circuit type
    particles_per_wire = cfg.model.particles_per_wire
    if cfg.model.which_circuit == "multiuploading":
        num_particles_needed = cfg.model.wires * particles_per_wire
    else:
        # Default, new, and reuploading circuits use one particle per wire
        num_particles_needed = cfg.model.wires
    
    # Dynamically add the calculated value to the config
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.model.num_particles_needed = num_particles_needed
    OmegaConf.set_struct(cfg, True)   # Re-enable struct mode
    
    # Validate multiuploading circuit configuration against available data
    if cfg.model.which_circuit == "multiuploading":
        total_particles_needed = num_particles_needed
        
        # Check available particles in data (peek at training data)
        try:
            with h5py.File(cfg.data.data_dir, "r") as f:
                available_particles = f["jetConstituentsList"].shape[1]
        except Exception as e:
            print(f"Warning: Could not read data file {cfg.data.data_dir} for validation: {e}")
            print("Skipping multiuploading circuit data compatibility check")
        else:
            if total_particles_needed > available_particles:
                print(f"ERROR: Multiuploading encoding configuration is incompatible with data!", flush=True)
                print(f"Configuration requires: {total_particles_needed} particles ({cfg.model.wires} wires × {particles_per_wire} particles_per_wire)", flush=True)
                print(f"Data provides: {available_particles} particles per jet", flush=True)
                print(f"", flush=True)
                print(f"Available options:", flush=True)
                print(f"  1. Reduce particles_per_wire to {available_particles // cfg.model.wires} or less", flush=True)
                print(f"  2. Reduce wires to {available_particles // particles_per_wire} or less", flush=True)
                print(f"  3. Use data with at least {total_particles_needed} particles per jet", flush=True)
                print(f"  4. Switch to 'new' or 'default' circuit which uses only {cfg.model.wires} particles", flush=True)
                raise ValueError("Multiuploading encoding configuration incompatible with data dimensions")
    
    # Adjust val_jets and test_jets to be divisible by batch_size and below limits
    max_val_jets = 20000
    max_test_jets = 10000

    original_val_jets = cfg.data.val_jets
    cfg.data.val_jets = min(cfg.data.val_jets, max_val_jets)
    cfg.data.val_jets = (cfg.data.val_jets // cfg.training.batch_size) * cfg.training.batch_size
    if original_val_jets != cfg.data.val_jets:
        print(f"Adjusting val_jets from {original_val_jets} to {cfg.data.val_jets} to be divisible by batch_size {cfg.training.batch_size} and below {max_val_jets}", flush=True)

    original_test_jets = cfg.data.test_jets
    cfg.data.test_jets = min(cfg.data.test_jets, max_test_jets)
    cfg.data.test_jets = (cfg.data.test_jets // cfg.training.batch_size) * cfg.training.batch_size
    if original_test_jets != cfg.data.test_jets:
        print(f"Adjusting test_jets from {original_test_jets} to {cfg.data.test_jets} to be divisible by batch_size {cfg.training.batch_size} and below {max_test_jets}", flush=True)
    
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
    print(f"Photon modes: {cfg.model.photon_modes}", flush=True)
    # print(f"Layers: {cfg.model.layers}", flush=True)
    print(f"Epochs: {cfg.training.epochs}", flush=True)
    print(f"Learning rate: {cfg.training.learning_rate}", flush=True)
    print(f"Batch size: {cfg.training.batch_size}", flush=True)
    print(f"Loss function: {cfg.training.loss_fn}", flush=True)
    if cfg.training.tanh:
        print("\t Tanh activation: True", flush=True)
    else:
        print("\t Sigmoid activation with trainable bias", flush=True)
    print(f"Which circuit: {cfg.model.which_circuit}", flush=True)
    if cfg.model.which_circuit == "multiuploading":
        print(f"Particles per wire: {cfg.model.particles_per_wire}", flush=True)
        print(f"Particle mapping: {cfg.model.particle_mapping}", flush=True)
    if cfg.model.which_circuit == "reuploading":
        print(f"Reuploads per wire: {cfg.model.reuploads_per_wire}", flush=True)
    print(f"Train jets: {cfg.data.train_jets}", flush=True)
    print(f"Validation jets: {cfg.data.val_jets}", flush=True)
    print(f"Test jets: {cfg.data.test_jets}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print("------------------------------------", flush=True)
