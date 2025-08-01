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
        f.write("# Format: jet_index, true_label, predicted_prob, predicted_class, jet_pt, [particle features...]\n")
        f.write("# Particle features: eta, phi, pt (for each particle used in the model)\n")
        f.write(f"# Model used {cfg.model.num_particles_needed} particles per jet\n")
        f.write(f"# Circuit type: {cfg.model.which_circuit}\n")
        f.write("\n")
        
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
            
            # Flatten particle features for output
            particle_features_str = ""
            for p in range(cfg.model.num_particles_needed):
                eta, phi, pt = jet_features[p, :]
                particle_features_str += f" {eta:.6f} {phi:.6f} {pt:.6f}"
            
            f.write(f"{jet_idx} {true_label:.0f} {pred_prob:.6f} {pred_class} {jet_pt_val:.6f}{particle_features_str}\n")
