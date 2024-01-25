import numpy as np
import tensorflow as tf


@tf.function
def get_model_gradients(model, loss_fn, X, y):
    '''
    Calculate model gradients with respect to weights

    Args:
        model (keras.model): Input model
        loss_fn (keras.losses.Loss): Model loss function
        X (np.array): Input data
        y (np.array): Output data

    Returns:
        grads (dict): Per-layer gradients of loss with respect to weights
    '''
    grads = {}
    # While persistent GradientTape slows down execution,
    # Is faster than performing forward pass and non-persistent for every layer
    with tf.GradientTape(persistent=True) as tape:
        output = model(X, training=True)
        loss_value = loss_fn(y, output)

        for layer in model.layers:
            if len(layer.trainable_weights) > 0:
                grads[layer.name] = tape.gradient(loss_value, layer.kernel)

    return grads


@tf.function
def get_model_hessians(model, loss_fn, X, y):
    '''
    Calculate the second derivatives of the loss with repsect to model weights.

    Note, only diagonal elements of the Hessian are computed.

    Args:
        model (keras.model): Input model
        loss_fn (keras.losses.Loss): Model loss function
        X (np.array): Input data
        y (np.array): Output data

    Returns:
        grads (dict): Per-layer second derivatives of loss with respect to weights
    '''
    grads = {}
    with tf.GradientTape(persistent=True) as tape:
        output = model(X, training=False)
        loss_value = loss_fn(y, output)

        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                grads[layer.name] = tape.gradient(tape.gradient(loss_value, layer.kernel), layer.kernel)

    return grads


def get_model_sparsity(model):
    '''
    Calculate total and per-layer model sparsity

    Args:
        - model (keras.model): Model to be evaluated

    Returns:
        tuple containing

        - sparsity (float): Model sparsity, as a percentage of zero weights w.r.t to total number of model weights
        - layers (dict): Key-value dictionary; each key is a layer name and the associated value is the layer's sparsity

    TODO - Extend support for recurrent layers (reccurent_kernel)
    '''

    total_weights = 0
    zero_weights = 0
    layer_sparsity = {}

    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.get_weights()[0].flatten()
            total_weights = total_weights + len(weights)
            zero_weights = zero_weights + len(weights) - np.count_nonzero(weights)
            layer_sparsity[layer.name] = 1.0 - np.count_nonzero(weights) / len(weights)

    try:
        return zero_weights / total_weights, layer_sparsity
    except ZeroDivisionError:
        return 0.0, layer_sparsity


# TODO - Does this work for non-linear models (e.g. skip connections) ?
def get_last_layer_with_weights(model):
    '''
    Finds the last layer with weights

    The last layer with weights determined the output shape, so, pruning is sometimes not applicable to it.
    As an example, consider a network with 16 - 32 - 5 neurons - the last layer's neuron (5) cannot be removed
    since they map to the data labels

    Args:
        model (keras.model): Input model

    Returns:
        idx (int): Index location of last layer with params
    '''
    for idx, layer in reversed(list(enumerate(model.layers))):
        if hasattr(layer, 'kernel'):
            return idx
    return len(model.layers)
