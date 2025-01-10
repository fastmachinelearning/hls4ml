import numpy as np
import pytest
from qkeras import QConv2D, QDense
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.masking import get_model_masks
from hls4ml.optimization.dsp_aware_pruning.objectives import ParameterEstimator

'''
In all the tests, an artifical network with one Dense/Conv2D layer and pre-determined weights is created
Then, the tests assert zeros occur in the correct places, based on the masking structure (unstructured, block etc.)
Furthermore, tests assert the masks are binary, so only zeros and ones occur
Masking is such that:
        * non_zero_params <= (1 - sparsity) * total_params OR
        * zero_params > sparsity * total_params
Since the targetted objective is ParameterEstimator, weight sharing is not suitable [does not decrease the number of weights]
Therefore, all the test verify offsets are zero
'''
sparsity = 0.33
local_masking = [True, False]
dense_layers = [Dense, QDense]
conv2d_layers = [Conv2D, QConv2D]


# Create a Dense layer with artificial weights, so that (1, 1) and (2, 3) (matrix indexing) are pruned
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_dense_masking_unstructured(local_masking, dense):
    weight_shape = (2, 3)
    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()
    weights[0][0, 0] = 1e-6
    weights[0][1, 2] = 1e-6
    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.UNSTRUCTURED

    # 33% sparsity - zero 2 out of 6 blocks with lowest norm [0.33 * 6 = 1.98 -> next largest int is 2]
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array([[0, 0], [1, 2]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['dense'] != 0), axis=1)

    assert not np.any(offsets['dense'])
    assert not np.any(masks['dense'][zeros[:, 0], zeros[:, 1]])
    assert (weight_shape[0] * weight_shape[1]) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Dense layer with artificial weights, so that the 1st and 3rd column (neuron) are pruned
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_dense_masking_structured(local_masking, dense):
    weight_shape = (3, 6)
    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()
    weights[0][:, 0] = 1e-6
    weights[0][:, 2] = 1e-6
    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.STRUCTURED

    # 33% sparsity - zero 2 out of 6 blocks with lowest norm [0.33 * 6 = 1.98 -> next largest int is 2]
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],  # First neuron
            [0, 2],
            [1, 2],
            [2, 2],  # Third neuron
        ],
        dtype=np.int32,
    )
    nonzeros = np.stack(np.where(masks['dense'] != 0), axis=1)

    assert not np.any(offsets['dense'])
    assert not np.any(masks['dense'][zeros[:, 0], zeros[:, 1]])
    assert (weight_shape[0] * weight_shape[1]) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Dense layer with artificial weights, so that some patterns are pruned
# Set pattern offset to 4, which is equivalent to RF = 3 [4 * 3 / 4]
# In this case consecutive patterns are one, so pruning per DSP block
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_dense_masking_pattern(local_masking, dense):
    weight_shape = (3, 4)
    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()

    # Set 1st block low
    weights[0][0, 0] = 1e-6
    weights[0][1, 0] = 1e-6
    weights[0][2, 0] = 1e-6

    # Set 3rd block low
    weights[0][0, 2] = 1e-6
    weights[0][1, 2] = 1e-6
    weights[0][2, 2] = 1e-6

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.PATTERN
    model_attributes['dense'].optimization_attributes.pattern_offset = 4
    model_attributes['dense'].optimization_attributes.consecutive_patterns = 1

    # 33% sparsity - zero 4 from 12 weights, group by pattern [0.33 * 12 = 3.96] - so will select 2 patterns, 6 weights (>=)
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array([[0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['dense'] != 0), axis=1)

    assert not np.any(offsets['dense'])
    assert not np.any(masks['dense'][zeros[:, 0], zeros[:, 1]])
    assert (weight_shape[0] * weight_shape[1]) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Dense layer with artificial weights, so that the 1st and 4th block are pruned
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
@pytest.mark.skip(
    reason='Currently disabled as no benefits from block pruning are achieved for hls4ml.'
)  # TODO - Enable when fully tested
def test_dense_masking_block(local_masking, dense):
    weight_shape = (4, 6)
    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()

    # Set 1st block low
    weights[0][0, 0] = 1e-6
    weights[0][0, 1] = 1e-6
    weights[0][1, 0] = 1e-6
    weights[0][1, 2] = 1e-6

    # Set 4th block low
    weights[0][2, 2] = 1e-6
    weights[0][2, 3] = 1e-6
    weights[0][3, 2] = 1e-6
    weights[0][3, 3] = 1e-6

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.BLOCK
    model_attributes['dense'].optimization_attributes.block_shape = (2, 2)

    # 33% sparsity - zero 2 out of 6 blocks with lowest norm
    # The first block is the smallest, the fourth block is set to zero
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['dense'] != 0), axis=1)

    assert not np.any(offsets['dense'])
    assert not np.any(masks['dense'][zeros[:, 0], zeros[:, 1]])
    assert (weight_shape[0] * weight_shape[1]) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Conv2D layer with artificial weights and mask some small weights
# Target sparsity is 0.33, so there should be <= (1 - 0.33) * 16 = 10.72 non-zero params
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('conv2d', conv2d_layers)
def test_conv2d_masking_unstructured(local_masking, conv2d):
    filt_width = 2
    filt_height = 2
    n_channels = 2
    n_filters = 2

    model = Sequential()
    model.add(conv2d(n_filters, input_shape=(8, 8, n_channels), kernel_size=(filt_width, filt_height), name='conv2d'))
    model.add(Flatten())
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()
    weights[0][0, 0, 0, 0] = 1e-6
    weights[0][1, 0, 0, 0] = 1e-6
    weights[0][0, 1, 0, 0] = 1e-6
    weights[0][0, 0, 1, 0] = 1e-6
    weights[0][0, 0, 0, 1] = 1e-6
    weights[0][1, 1, 1, 1] = 1e-6
    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['conv2d'].optimizable = True
    model_attributes['conv2d'].optimization_attributes.pruning = True
    model_attributes['conv2d'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.UNSTRUCTURED

    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['conv2d'] != 0), axis=1)

    assert not np.any(offsets['conv2d'])
    assert not np.any(masks['conv2d'][zeros[:, 0], zeros[:, 1], zeros[:, 2], zeros[:, 3]])
    assert (filt_width * filt_height * n_channels * n_filters) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Conv2D layer with artificial weights, so that second and last filter are pruned
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('conv2d', conv2d_layers)
def test_conv2d_masking_structured(local_masking, conv2d):
    filt_width = 3
    filt_height = 3
    n_channels = 4
    n_filters = 6

    model = Sequential()
    model.add(conv2d(n_filters, input_shape=(8, 8, n_channels), kernel_size=(filt_width, filt_height), name='conv2d'))
    model.add(Flatten())
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()
    weights[0][:, :, :, 1] = 1e-3
    weights[0][:, :, :, 5] = 1e-3
    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['conv2d'].optimizable = True
    model_attributes['conv2d'].optimization_attributes.pruning = True
    model_attributes['conv2d'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.STRUCTURED

    # 33% sparsity - zero 2 out of 6 filters with lowest norm
    # Generate all possible combinations of width and height pixels with channel using np.meshgrid()
    # This represents all the positions for a single filter; then append filter position to the last columns
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    width_pixels = np.array(range(0, filt_width))
    height_pixels = np.array(range(0, filt_height))
    channels = np.array(range(0, n_channels))
    combinations = np.array(np.meshgrid(width_pixels, height_pixels, channels)).T.reshape(-1, 3)
    zeros = np.array(
        np.append(combinations, np.full((filt_width * filt_height * n_channels, 1), 1), axis=1).tolist()
        + np.append(combinations, np.full((filt_width * filt_height * n_channels, 1), 5), axis=1).tolist(),
        dtype=np.int32,
    )
    nonzeros = np.stack(np.where(masks['conv2d'] != 0), axis=1)

    assert not np.any(offsets['conv2d'])
    assert not np.any(masks['conv2d'][zeros[:, 0], zeros[:, 1], zeros[:, 2], zeros[:, 3]])
    assert (filt_width * filt_height * n_channels * n_filters) == (zeros.shape[0] + nonzeros.shape[0])


# Create a Conv2D layer with artificial weights, so that the first and second pattern are pruned
# Set pattern offset to 4, which is equivalent to RF = 2 [2 * 2 * 2 / 4]
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('conv2d', conv2d_layers)
def test_conv2d_masking_pattern(local_masking, conv2d):
    filt_width = 2
    filt_height = 1
    n_channels = 2
    n_filters = 2

    model = Sequential()
    model.add(conv2d(n_filters, input_shape=(8, 8, n_channels), kernel_size=(filt_width, filt_height), name='conv2d'))
    model.add(Flatten())
    model.add(Dense(1, name='out'))

    weights = model.layers[0].get_weights()

    # Set the first DSP block to be approximately zero
    weights[0][0, 0, 0, 0] = 1e-6
    weights[0][0, 0, 1, 0] = 1e-6

    # Set the third DSP block to be approximately zero
    weights[0][0, 0, 0, 1] = 1e-6
    weights[0][0, 0, 1, 1] = 1e-6

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['conv2d'].optimizable = True
    model_attributes['conv2d'].optimization_attributes.pruning = True
    model_attributes['conv2d'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.PATTERN
    model_attributes['conv2d'].optimization_attributes.pattern_offset = 4

    # 33% sparsity - zero out the two of the lowest groups
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    print(masks['conv2d'].shape)
    print(weights[0].shape)
    zeros = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['conv2d'] != 0), axis=1)

    assert not np.any(offsets['conv2d'])
    assert not np.any(masks['conv2d'][zeros[:, 0], zeros[:, 1], zeros[:, 2], zeros[:, 3]])
    assert (filt_width * filt_height * n_channels * n_filters) == (zeros.shape[0] + nonzeros.shape[0])


# Block pruning is only allowed for 2-dimensional matrices, so assert a correct exception is raised when pruning with Conv2D
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('conv2d', conv2d_layers)
def test_conv2d_block_masking_raises_exception(local_masking, conv2d):
    model = Sequential()
    model.add(conv2d(4, input_shape=(8, 8, 3), kernel_size=(3, 3), name='conv2d'))
    model.add(Flatten())
    model.add(Dense(1, name='out'))

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['conv2d'].optimizable = True
    model_attributes['conv2d'].optimization_attributes.pruning = True
    model_attributes['conv2d'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.BLOCK

    try:
        get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    except Exception:
        assert True
        return
    assert not True


# Test edge cases: 0% and 100% sparsity
# Test 50% sparsity with two layers
@pytest.mark.parametrize('s', [0, 0.5, 1])
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize(
    'type',
    [
        SUPPORTED_STRUCTURES.UNSTRUCTURED,
        SUPPORTED_STRUCTURES.STRUCTURED,
        SUPPORTED_STRUCTURES.PATTERN,
        SUPPORTED_STRUCTURES.BLOCK,
    ],
)
def test_multi_layer_masking(s, local_masking, type):
    dense_units = 16
    conv_filters = 6
    conv_channels = 4
    conv_shape = (2, 2)  # Using (2, 2) instead of (3, 3) as it's an even number of weights
    input_shape = (8, 8)

    # Simple model, Conv2D weight shape (2, 2, 4, 6) and Dense weight shape (384, 16)
    model = Sequential()
    model.add(
        Conv2D(
            conv_filters,
            input_shape=(*input_shape, conv_channels),
            kernel_size=conv_shape,
            name='conv2d',
            padding='same',
            kernel_initializer='ones',
        )
    )
    model.add(Flatten())
    model.add(Dense(dense_units, name='dense', kernel_initializer='ones'))

    # Make 'dense' and 'conv2d' optimizable
    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = type
    model_attributes['dense'].optimization_attributes.pattern_offset = 1024  # Equivalent to RF = 6 (384 * 16 / 1024)

    model_attributes['conv2d'].optimizable = True
    model_attributes['conv2d'].optimization_attributes.pruning = True
    model_attributes['conv2d'].optimization_attributes.structure_type = (
        type if type != SUPPORTED_STRUCTURES.BLOCK else SUPPORTED_STRUCTURES.UNSTRUCTURED
    )
    model_attributes['conv2d'].optimization_attributes.pattern_offset = 4  # Equivalent to RF = 4 (2 * 2 * 4 * 6 / 4)

    masks, offsets = get_model_masks(model, model_attributes, s, ParameterEstimator, metric='l1', local=local_masking)
    if s == 1:  # 100% sparsity - all masks are zero
        print(np.count_nonzero(masks['dense'].flatten()))
        assert not np.any(masks['dense'])
        assert not np.any(masks['conv2d'])
    elif s == 0.5:
        conv2d_weights = conv_channels * conv_filters * np.prod(conv_shape)
        dense_weights = dense_units * np.prod(input_shape) * conv_filters
        if local_masking:
            assert np.count_nonzero(masks['conv2d']) == int((1 - s) * conv2d_weights)
            assert np.count_nonzero(masks['dense']) == int((1 - s) * dense_weights)
        else:
            # Less than or equal to, since Knapsack problem imposes a hard constrain on the active resources (ones)
            assert np.count_nonzero(masks['conv2d']) + np.count_nonzero(masks['dense']) <= int(
                (1 - s) * (conv2d_weights + dense_weights)
            )
    elif s == 0:  # 0% sparsity - all masks are one
        assert np.all(masks['dense'])
        assert np.all(masks['conv2d'])

    assert not np.any(offsets['dense'])
    assert not np.any(offsets['conv2d'])


# Create a Dense layer with artificial weights, so that some consecutive patterns are pruned
# Set consecutive patterns to 2, so that the 1st block [1st and 2nd pattern] are pruned
@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_consecutive_pattern_masking(local_masking, dense):
    weight_shape = (3, 4)
    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    model.add(Flatten())
    model.add(Dense(1, name='out'))
    weights = model.layers[0].get_weights()

    weights[0] = np.arange(np.prod(weight_shape)).reshape(weight_shape)

    # Set 1st and 2nd pattern low
    weights[0][0, 0] = 1e-6
    weights[0][1, 0] = 1e-6
    weights[0][2, 0] = 1e-6
    weights[0][0, 1] = 1e-4
    weights[0][1, 1] = 1e-4
    weights[0][2, 1] = 1e-4

    # Set 4th pattern lower than second
    # This pattern should still remain unmasked [even if it has a lower value than the 2nd pattern],
    # As its neigbouring block has a larger value than the 2nd pattern
    weights[0][0, 3] = 1e-6
    weights[0][1, 3] = 1e-6
    weights[0][2, 3] = 1e-6

    print(weights)
    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.PATTERN
    model_attributes['dense'].optimization_attributes.pattern_offset = 4
    model_attributes['dense'].optimization_attributes.consecutive_patterns = 2

    # 33% sparsity - zero 4 out of 12 weight, group by pattern [0.33 * 12 = 3.96]
    masks, offsets = get_model_masks(model, model_attributes, sparsity, ParameterEstimator, metric='l1', local=local_masking)
    zeros = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.int32)
    nonzeros = np.stack(np.where(masks['dense'] != 0), axis=1)

    assert not np.any(offsets['dense'])
    assert not np.any(masks['dense'][zeros[:, 0], zeros[:, 1]])
    assert (weight_shape[0] * weight_shape[1]) == (zeros.shape[0] + nonzeros.shape[0])
