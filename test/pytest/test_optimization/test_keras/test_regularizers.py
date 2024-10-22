import numpy as np
import pytest
import tensorflow as tf
from qkeras import QConv2D, QDense
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.builder import remove_custom_regularizers
from hls4ml.optimization.dsp_aware_pruning.keras.regularizers import Conv2DRegularizer, DenseRegularizer

# Constants
pattern_offset = 4
consecutive_patterns = 2
block_shape = (4, 4)

dense_layers = [Dense, QDense]
conv2d_layers = [Conv2D, QConv2D]


# Sets the loss due to data to zero; train model only on regularization loss
def zero_loss(y_true, y_pred):
    return tf.reduce_mean(0 * tf.square(y_true - y_pred), axis=-1)


# Helper function, calculates the group norm and variance for a single layer
def get_norm_and_variance(weights, structure_type, layer='dense'):
    if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
        norm = np.linalg.norm(weights.flatten(), ord=1)
        var = np.var(weights)
        return norm, var

    if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
        if layer == 'conv2d':
            norm = np.linalg.norm(np.sum(np.linalg.norm(weights, axis=(0, 1), ord='fro'), axis=0), ord=1)
            var = np.linalg.norm(np.var(weights, axis=(0, 1, 2)), ord=1)
        else:
            norm = np.linalg.norm(np.linalg.norm(weights, axis=0, ord=2), ord=1)
            var = np.linalg.norm(np.var(weights, axis=0), ord=1)

        return norm, var

    if structure_type == SUPPORTED_STRUCTURES.PATTERN:
        number_of_patterns = np.prod(weights.shape) // pattern_offset
        target_shape = (number_of_patterns, pattern_offset)
        reshaped = np.reshape(weights, target_shape)
        total_blocks = pattern_offset // consecutive_patterns
        blocks = np.reshape(
            tf.image.extract_patches(
                np.expand_dims(np.expand_dims(reshaped, 2), 0),
                [1, number_of_patterns, consecutive_patterns, 1],
                [1, number_of_patterns, consecutive_patterns, 1],
                [1, 1, 1, 1],
                'SAME',
            ).numpy(),
            (total_blocks, number_of_patterns * consecutive_patterns),
        )
        norm = np.linalg.norm(np.linalg.norm(blocks, axis=1, ord=2), ord=1)
        var = np.linalg.norm(np.var(blocks, axis=1), ord=1)
        return norm, var

    if structure_type == SUPPORTED_STRUCTURES.BLOCK:
        total_blocks = (weights.shape[0] * weights.shape[1]) // (block_shape[0] * block_shape[1])
        blocks = np.reshape(
            tf.image.extract_patches(
                np.expand_dims(np.expand_dims(weights, 2), 0),
                [1, block_shape[0], block_shape[1], 1],
                [1, block_shape[0], block_shape[1], 1],
                [1, 1, 1, 1],
                'SAME',
            ).numpy(),
            (total_blocks, block_shape[0] * block_shape[1]),
        )

        norm = np.linalg.norm(np.linalg.norm(blocks, axis=1, ord=2), ord=1)
        var = np.linalg.norm(np.var(blocks, axis=1), ord=1)
        return norm, var


@pytest.mark.parametrize('dense', dense_layers)
@pytest.mark.parametrize(
    'structure_type',
    [
        SUPPORTED_STRUCTURES.UNSTRUCTURED,
        SUPPORTED_STRUCTURES.STRUCTURED,
        SUPPORTED_STRUCTURES.PATTERN,
        SUPPORTED_STRUCTURES.BLOCK,
    ],
)
def test_dense_regularizer(structure_type, dense):
    epochs = 10
    data_points = 10
    input_shape = (32,)
    output_shape = (16,)
    X = np.random.rand(data_points, *input_shape)
    y = np.random.rand(data_points, *output_shape)
    w = np.random.rand(input_shape[0], output_shape[0])

    # First, fit a model without regularization
    model = Sequential()
    model.add(dense(output_shape[0], input_shape=input_shape))
    dense_weights = model.layers[0].get_weights()
    dense_weights[0] = w
    model.layers[0].set_weights(dense_weights)
    model.compile(loss=zero_loss, optimizer=Adam(1.0))
    model.fit(X, y, epochs=epochs)
    norm, var = get_norm_and_variance(model.layers[0].get_weights()[0], structure_type)

    # Now, fit a model with strong regularization, starting with the same initial weights
    dense_weights = model.layers[0].get_weights()
    dense_weights[0] = w
    model.layers[0].set_weights(dense_weights)
    regularizer = DenseRegularizer(
        alpha=0.5, beta=0.5, structure_type=structure_type, pattern_offset=pattern_offset, block_shape=block_shape
    )
    model.layers[0].add_loss(lambda layer=model.layers[0]: regularizer(layer.kernel))
    model.compile(loss=zero_loss, optimizer=Adam(1.0))
    model.fit(X, y, epochs=epochs)
    reg_norm, reg_var = get_norm_and_variance(model.layers[0].get_weights()[0], structure_type)

    # Verify regularization decreased weight magnitude and variance
    assert reg_norm < norm
    assert reg_var < var


@pytest.mark.parametrize('conv2d', conv2d_layers)
@pytest.mark.parametrize(
    'structure_type', [SUPPORTED_STRUCTURES.UNSTRUCTURED, SUPPORTED_STRUCTURES.STRUCTURED, SUPPORTED_STRUCTURES.PATTERN]
)
def test_conv2d_regularizer(structure_type, conv2d):
    epochs = 10
    data_points = 10
    input_shape = (16, 16, 3)
    num_filters = 4
    X = np.random.rand(data_points, *input_shape)
    y = np.random.rand(data_points, 1)

    # First, fit a model without regularization
    model = Sequential()
    model.add(conv2d(num_filters, (3, 3), input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    conv_weights = model.layers[0].get_weights()
    model.compile(loss=zero_loss, optimizer=Adam())
    model.fit(X, y, epochs=epochs, verbose=True)
    norm, var = get_norm_and_variance(model.layers[0].get_weights()[0], structure_type, 'conv2d')

    # Now, fit a model with strong regularization, starting with the same initial weights
    model.layers[0].set_weights(conv_weights)
    regularizer = Conv2DRegularizer(alpha=0.5, beta=0.5, structure_type=structure_type, pattern_offset=pattern_offset)
    model.layers[0].add_loss(lambda layer=model.layers[0]: regularizer(layer.kernel))
    model.compile(loss=zero_loss, optimizer=Adam())
    model.fit(X, y, epochs=epochs, verbose=True)
    reg_norm, reg_var = get_norm_and_variance(model.layers[0].get_weights()[0], structure_type, 'conv2d')

    # Verify regularization decreased weight magnitude and variance
    assert reg_norm < norm
    assert reg_var < var


def test_removal_of_custom_regularizer():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), input_shape=(16, 16, 3), kernel_regularizer=Conv2DRegularizer(1e-3)))
    model.add(Flatten())
    model.add(Dense(1, kernel_regularizer=DenseRegularizer(1e-3)))
    weights = model.get_weights()

    assert isinstance(model.layers[0].kernel_regularizer, Conv2DRegularizer)
    assert isinstance(model.layers[2].kernel_regularizer, DenseRegularizer)

    model = remove_custom_regularizers(model)

    assert model.layers[0].kernel_regularizer is None
    for i in range(len(weights)):
        assert np.all(weights[i] == model.get_weights()[i])
