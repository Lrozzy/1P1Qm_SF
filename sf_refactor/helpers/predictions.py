"""Utilities for saving model predictions and inputs."""

import os
import numpy as np


def save_test_predictions(cfg, run_name, prob_test, pred_test, labels_test, jet_pt_test, jets_test):
    """
    Save test predictions and inputs to a file.
    
    Args:
        cfg: Configuration object
        run_name: Name of the current run
        prob_test: Array of predicted probabilities
        pred_test: Array of predicted classes (binary)
        labels_test: Array of true labels
        jet_pt_test: Array of jet pT values
        jets_test: Array of jet particle features
    """
    predictions_path = os.path.join(cfg.data.save_dir, run_name, "test_predictions.txt")
    print(f"Saving test predictions to {predictions_path}...", flush=True)
    
    with open(predictions_path, "w") as f:
        f.write("# Test set predictions and inputs\n")
        f.write("# Format: CSV-like with named columns for easy parsing\n")
        f.write(f"# Model: {cfg.model.which_circuit} circuit with {cfg.model.num_particles_needed} particles per jet\n")
        f.write("#\n")
        f.write("# Column descriptions:\n")
        f.write("#   jet_idx: Index of jet in test set\n")
        f.write("#   true_label: Ground truth label (0 or 1)\n")
        f.write("#   pred_class: Predicted class (0 or 1)\n")
        f.write("#   pred_prob: Predicted probability\n")
        f.write("#   jet_pt: Jet transverse momentum\n")
        f.write("#   particle_N_eta/phi/pt: Features for particle N (eta, phi, pt)\n")
        f.write("#\n")
        
        # Write header
        header = "jet_idx,true_label,pred_class,pred_prob,jet_pt"
        for p in range(cfg.model.num_particles_needed):
            header += f",particle_{p+1}_eta,particle_{p+1}_phi,particle_{p+1}_pt"
        f.write(header + "\n")
        
        # Get the number of jets that were actually processed (may be less due to batching)
        num_processed_jets = len(prob_test)
        
        for i in range(num_processed_jets):
            jet_idx = i
            true_label = labels_test.numpy()[i]
            pred_prob = prob_test[i]
            pred_class = pred_test[i]
            jet_pt_val = jet_pt_test.numpy()[i]
            
            # Get particle features for this jet (limited to num_particles_needed)
            jet_features = jets_test.numpy()[i, :cfg.model.num_particles_needed, :]  # [particles, features]
            
            # Build CSV row
            row = f"{jet_idx},{true_label:.0f},{pred_class},{pred_prob:.6f},{jet_pt_val:.6f}"
            for p in range(cfg.model.num_particles_needed):
                eta, phi, pt = jet_features[p, :]
                row += f",{eta:.6f},{phi:.6f},{pt:.6f}"
            
            f.write(row + "\n")
