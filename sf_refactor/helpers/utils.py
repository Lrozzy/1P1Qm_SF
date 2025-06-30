import h5py
import numpy as np
import tensorflow as tf

# DATA IS ALREADY SORTED BY PT (highest to lowest)
def load_data(path, max_jets, wires):
    with h5py.File(path, "r") as f:
        jet_constituents = f["jetConstituentsList"][:max_jets, :wires, :]     # (N,4,3)
        truth_labels = f["truth_labels"][:max_jets].astype(np.float32)     # (N,)
    return tf.convert_to_tensor(jet_constituents, tf.float32), tf.convert_to_tensor(truth_labels, tf.float32)

def get_loss_fn(photons, label, shift_sigmoid=None, tanh = False, loss_type="bce", dim_cutoff=None):
    """
    Calculates loss and probabilities from photon counts.
    The logit is taken as the mean photon number across all modes.
    """
    # The logit is the mean of the photon counts over the modes for each item in the batch.
    logit = tf.reduce_mean(photons, axis=1) # axis=1 is the mode dimension

    if dim_cutoff is not None:
        # This normalization seems odd, but keeping it as it was in the original
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
            loss = tf.keras.losses.binary_crossentropy(label, prob, from_logits=False)
        else:
            # If using sigmoid, we can use the more numerically stable from_logits=True.
            logit_final = logit
            if shift_sigmoid is not None:
                logit_final = logit - shift_sigmoid
                
            # logit_final = tf.nn.relu(logit_final)
            loss = tf.keras.losses.binary_crossentropy(label, logit_final, from_logits=True)
            prob = tf.sigmoid(logit_final)
        
        return loss, prob

    else:
        raise ValueError(f"Unknown loss type: {loss_type} (use 'bce' or 'mse')")