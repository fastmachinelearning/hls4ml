import numpy as np
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning.keras.utils import get_last_layer_with_weights


def reduce_model(model):
    '''
    Function for removing zero neurons & filters from a model and rewiring the model graph
    This function is built on top of Keras Surgeon available at: https://github.com/BenWhetton/keras-surgeon
    Keras Surgeon is no longer under active development and does not work for TensorFlow 2.3+ and QKeras
    The baseline version was forked and updated, available at: https://github.com/fastmachinelearning/keras-surgeon

    IMPORTANT: To use this funcionality please install separately from the above GitHub.

    Args:
        model (keras.model): Input model

    Returns:
        reduced (keras.model): Modified model, with redundant structures removed

    '''
    try:
        from kerassurgeon import Surgeon
    except ModuleNotFoundError:
        raise Exception(
            'Keras Surgeon not installed. Unable to reduce model footprint '
            'Please install up-to-date Keras Surgeon compatible wit TensorFlow 2.3+ and QKeras '
            'Installation from git: https://github.com/fastmachinelearning/keras-surgeon'
        )

    # Initiate surgeon
    surgeon = Surgeon(model)

    # Iterate through layers and identify neurons (columns) and filters (tensors, W x H x C) to be removed
    last_idx = get_last_layer_with_weights(model)
    for idx, layer in enumerate(model.layers):
        # Last layer with weights cannot be removed, as it maps to data set labels
        if idx == last_idx:
            break

        # Currently supported Dense and Conv2D; these two can be combined in a single if-statement
        # Keras Surgeon has a full range of support for Conv1D / Conv3D, reucurrent etc. - might extend in the future
        if isinstance(layer, Dense):
            weights = layer.get_weights()[0]
            zeros = np.where(~weights.any(axis=0))[0].tolist()
            surgeon.add_job('delete_channels', layer, channels=zeros)

        elif isinstance(layer, Conv2D):
            weights = layer.get_weights()[0]
            zeros = np.where(~weights.reshape(-1, weights.shape[-1]).any(axis=0))[0].tolist()
            surgeon.add_job('delete_channels', layer, channels=zeros)

    # Reduce model
    reduced = surgeon.operate()

    # By default, Keras surgeon returns a Functional model
    # If the original was a Sequential, convert back
    is_sequential = model.__class__.__name__ == 'Sequential'
    if is_sequential:
        return Sequential(layers=reduced.layers)
    else:
        return reduced
