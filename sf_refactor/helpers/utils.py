import h5py
import numpy as np
import tensorflow as tf

# DATA IS ALREADY SORTED BY PT (highest to lowest)
def load_data(path, max_jets, num_particles):
    """
    Load jet data from HDF5 file.
    
    Args:
        path: Path to HDF5 file
        max_jets: Maximum number of jets to load
        num_particles: Number of particles to load per jet
    
    Returns:
        Tuple of (jet_constituents, truth_labels, jet_pt) as TensorFlow tensors
    """
    with h5py.File(path, "r") as f:
        available_particles = f["jetConstituentsList"].shape[1]
        
        if num_particles > available_particles:
            raise ValueError(
                f"Requested {num_particles} particles per jet "
                f"but data only has {available_particles} particles per jet"
            )
        
        jet_constituents = f["jetConstituentsList"][:max_jets, :num_particles, :]
        truth_labels = f["truth_labels"][:max_jets].astype(np.float32)
        
        # Load jet pT (first column of jetFeatures)
        jet_features = f["jetFeatures"][:max_jets, :]
        jet_pt = jet_features[:, 0].astype(np.float32)
    
    return (tf.convert_to_tensor(jet_constituents, tf.float32), 
            tf.convert_to_tensor(truth_labels, tf.float32),
            tf.convert_to_tensor(jet_pt, tf.float32))

def get_loss_fn(photons, label, bias=0.0, tanh = False, loss_type="bce", dim_cutoff=None):
    """
    Calculates loss and probabilities from photon counts.
    The logit is taken as the mean photon number across all modes.
    """
    # The logit is the mean of the photon counts over the modes for each item in the batch.
    logit = tf.reduce_mean(photons, axis=1) # axis=1 is the mode dimension

    if dim_cutoff is not None:
        logit /= dim_cutoff

    if loss_type.lower() == "mse":
        # For MSE, the output should be a probability.
        prob = tf.sigmoid(logit)
        loss = tf.keras.losses.mean_squared_error(label, prob)
        return loss, prob

    elif loss_type.lower() == "bce":
        # For BCE, we work with logits.
        if tanh:
            # If using tanh, the output is already in [-1, 1]. Scale to [0, 1] for probability.
            # The loss function should not see logits in this case.
            prob = (tf.tanh(logit) + 1) / 2.0
            # Manually compute binary cross-entropy to ensure per-sample losses
            epsilon = 1e-7  # Small value to prevent log(0)
            prob_clipped = tf.clip_by_value(prob, epsilon, 1 - epsilon)
            loss = -(label * tf.math.log(prob_clipped) + (1 - label) * tf.math.log(1 - prob_clipped))
        else:
            # If using sigmoid, we can use the more numerically stable from_logits=True.
            logit_final = logit - bias

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_final)
            prob = tf.sigmoid(logit_final)
        
        return loss, prob

    else:
        raise ValueError(f"Unknown loss type: {loss_type} (use 'bce' or 'mse')")