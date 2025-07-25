import h5py
import numpy as np
import tensorflow as tf

# DATA IS ALREADY SORTED BY PT (highest to lowest)
def load_data(path, max_jets, wires):
    with h5py.File(path, "r") as f:
        jet_constituents = f["jetConstituentsList"][:max_jets, :wires, :]     # (N,4,3)
        truth_labels = f["truth_labels"][:max_jets].astype(np.float32)     # (N,)
    return tf.convert_to_tensor(jet_constituents, tf.float32), tf.convert_to_tensor(truth_labels, tf.float32)

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