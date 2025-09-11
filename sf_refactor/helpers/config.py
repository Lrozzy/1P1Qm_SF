"""Configuration management utilities for Hydra-based SF training.

This module centralizes save path creation. It now supports:
 - Separate base save dirs for classifier and autoencoder (from config)
 - Day grouping (YYYY_MM_DD) under the base dir
 - Time-only run directory names (HH_MM or HH_MM_SS when needed)
All code should reference cfg.runtime.run_dir as the concrete run output dir.
"""

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


def _get_base_save_dir(cfg: DictConfig, exp_type: str) -> str:
    """Resolve base save dir from config for given experiment type."""
    if exp_type == "autoencoder":
        return getattr(cfg.data, "autoencoder_save_dir", cfg.data.save_dir)
    return getattr(cfg.data, "classifier_save_dir", cfg.data.save_dir)


def setup_run_name(cfg: DictConfig) -> str:
    """Return a time-only run name with optional user suffix.

    Format: HH_MM or HH_MM_<suffix>. Seconds will be automatically added by
    create_run_directory() if needed to avoid collisions.
    """
    time_str = datetime.now().strftime("%H_%M")
    if cfg.runtime.run_name:
        return f"{time_str}_{cfg.runtime.run_name}"
    return time_str


def create_run_directory(cfg: DictConfig, exp_type: str) -> str:
    """Create and register the concrete output directory for this run.

    Directory layout:
      <base>/<YYYY_MM_DD>/<HH_MM[_SS][_suffix]>

    - base is selected by exp_type via config:
        data.classifier_save_dir or data.autoencoder_save_dir
    - Also sets cfg.runtime.run_dir to this absolute path for global use.
    - Returns the final run_name (time portion incl. optional seconds/suffix).
    """
    # Day folder
    day_str = datetime.now().strftime("%Y_%m_%d")
    base_dir = _get_base_save_dir(cfg, exp_type)
    day_dir = os.path.join(base_dir, day_str)
    if not getattr(cfg.runtime, 'cli_test', False):
        os.makedirs(day_dir, exist_ok=True)

    # Start from time-only name (with optional user suffix)
    base_run_name = setup_run_name(cfg)  # HH_MM[_suffix]
    run_name = base_run_name

    # Ensure uniqueness within the day; if exists, add seconds; if still, add _1, _2...
    candidate = os.path.join(day_dir, run_name)
    if os.path.exists(candidate):
        # Add seconds to differentiate runs within the same minute
        sec_str = datetime.now().strftime("%S")
        run_name = f"{base_run_name}_{sec_str}"
        candidate = os.path.join(day_dir, run_name)
        i = 1
        while os.path.exists(candidate):
            run_name = f"{base_run_name}_{sec_str}_{i}"
            candidate = os.path.join(day_dir, run_name)
            i += 1

    # Create concrete run directory and attach to cfg for global reference
    if not getattr(cfg.runtime, 'cli_test', False):
        os.makedirs(candidate, exist_ok=True)

    # Store canonical references for downstream helpers
    OmegaConf.set_struct(cfg, False)
    cfg.runtime.run_dir = candidate
    cfg.runtime.run_name = run_name
    cfg.runtime.exp_type = exp_type
    # Maintain old save_dir for backward compatibility/printing
    cfg.data.save_dir = _get_base_save_dir(cfg, exp_type)
    OmegaConf.set_struct(cfg, True)

    return run_name


def save_config_to_file(cfg: DictConfig, run_name: str, seed: int, exclude_loss: bool = False):
    """Save configuration parameters to a file.
    Args:
        cfg: OmegaConf config
        run_name: directory/run identifier
        seed: random seed to record
        exclude_loss: when True, do not persist training.loss_fn in params.txt
    """
    if not cfg.runtime.cli_test:
        # Use resolved run_dir (single source of truth)
        run_dir = getattr(cfg.runtime, "run_dir", os.path.join(cfg.data.save_dir, run_name))
        os.makedirs(run_dir, exist_ok=True)

        # Get current date and time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert config to dict and add runtime info
        params = OmegaConf.to_container(cfg, resolve=True)
        params['run_start_time'] = now
        params['seed'] = seed
        params['run_name'] = run_name
        params['run_dir'] = getattr(cfg.runtime, 'run_dir', '')
        # Include exp_type only if explicitly set by the script
        if hasattr(cfg.runtime, 'exp_type'):
            params['exp_type'] = cfg.runtime.exp_type

        # Flatten the nested config for easier reading
        flat_params = {}
        for section, values in params.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    full_key = f"{section}.{key}"
                    if exclude_loss and full_key == "training.loss_fn":
                        continue
                    flat_params[full_key] = value
            else:
                flat_params[section] = values

        params_path = os.path.join(run_dir, "params.txt")
        with open(params_path, "w") as f:
            for k, v in flat_params.items():
                f.write(f"{k}: {v}\n")


def print_config(cfg: DictConfig, run_name: str, seed: int, hide_loss: bool = False):
    """Print configuration parameters.
    Args:
        hide_loss: when True, do not print the training loss function line
    """
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
    if not hide_loss:
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
