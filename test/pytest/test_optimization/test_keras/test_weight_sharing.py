import numpy as np
import pytest
from qkeras import QDense
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.masking import get_model_masks
from hls4ml.optimization.dsp_aware_pruning.objectives import ObjectiveEstimator

# Similar tests in test_masking.py, weight sharing instead of pruning
sparsity = 0.33
local_masking = [True, False]
dense_layers = [Dense, QDense]

'''
A mock objective class for weight sharing
When a group of weights is quantized to the mean value, resource savings are equal to the number of weights quantized
This is similar to ParameterEstimator, but instead of pruning, weight sharing is performed and
No savings are incurred with unstructured type (unstructured weight sharing doesn't make sense)
'''


class MockWeightSharingEstimator(ObjectiveEstimator):
    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]
        else:
            return [np.prod(layer_attributes.weight_shape)]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type

        if layer_attributes.optimization_attributes.weight_sharing:
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                return [0]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                if 'Dense' in layer_attributes.layer_type.__name__:
                    return [layer_attributes.weight_shape[1]]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                number_of_patterns = (
                    np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
                )
                return [number_of_patterns * layer_attributes.optimization_attributes.consecutive_patterns]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                return [np.prod(layer_attributes.optimization_attributes.block_shape)]
        return [0]


@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_weight_sharing_structured(local_masking, dense):
    weight_shape = (4, 3)

    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    weights = model.layers[0].get_weights()

    weights[0][:, 1] = 0.5
    weights[0][0, 1] -= 1e-4
    weights[0][2, 1] += 1e-4

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = False
    model_attributes['dense'].optimization_attributes.weight_sharing = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.STRUCTURED

    masks, offsets = get_model_masks(
        model, model_attributes, sparsity, MockWeightSharingEstimator, metric='l1', local=local_masking
    )
    frozen = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32)

    assert not np.any(masks['dense'][frozen[:, 0], frozen[:, 1]])
    assert np.all(offsets['dense'][frozen[:, 0], frozen[:, 1]] == 0.5)


@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_weight_sharing_pattern(local_masking, dense):
    weight_shape = (3, 4)

    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    weights = model.layers[0].get_weights()
    weights[0][0, 0] = 0.5 + 1e-4
    weights[0][1, 0] = 0.5 - 1e-4
    weights[0][2, 0] = 0.5

    weights[0][0, 2] = 0.5 + 1e-4
    weights[0][1, 2] = 0.5 - 1e-4
    weights[0][2, 2] = 0.5

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = False
    model_attributes['dense'].optimization_attributes.weight_sharing = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.PATTERN
    model_attributes['dense'].optimization_attributes.pattern_offset = 4
    model_attributes['dense'].optimization_attributes.consecutive_patterns = 1

    masks, offsets = get_model_masks(
        model, model_attributes, sparsity, MockWeightSharingEstimator, metric='l1', local=local_masking
    )
    frozen = np.array([[0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]], dtype=np.int32)

    assert not np.any(masks['dense'][frozen[:, 0], frozen[:, 1]])
    assert np.all(offsets['dense'][frozen[:, 0], frozen[:, 1]] == 0.5)


@pytest.mark.parametrize('local_masking', local_masking)
@pytest.mark.parametrize('dense', dense_layers)
def test_weight_sharing_block(local_masking, dense):
    weight_shape = (4, 6)

    model = Sequential()
    model.add(dense(weight_shape[1], input_shape=(weight_shape[0],), name='dense'))
    weights = model.layers[0].get_weights()

    weights[0][0, 0] = 0.5 + 1e-3
    weights[0][0, 1] = 0.5 + 1e-3
    weights[0][1, 0] = 0.5 - 1e-3
    weights[0][1, 1] = 0.5 - 1e-3

    weights[0][2, 2] = 0.5 + 1e-3
    weights[0][2, 3] = 0.5 + 1e-3
    weights[0][3, 2] = 0.5 - 1e-3
    weights[0][3, 3] = 0.5 - 1e-3

    model.layers[0].set_weights(weights)

    model_attributes = get_attributes_from_keras_model(model)
    model_attributes['dense'].optimizable = True
    model_attributes['dense'].optimization_attributes.pruning = False
    model_attributes['dense'].optimization_attributes.weight_sharing = True
    model_attributes['dense'].optimization_attributes.structure_type = SUPPORTED_STRUCTURES.BLOCK
    model_attributes['dense'].optimization_attributes.block_shape = (2, 2)

    masks, offsets = get_model_masks(
        model, model_attributes, sparsity, MockWeightSharingEstimator, metric='l1', local=local_masking
    )
    frozen = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]], dtype=np.int32)

    assert not np.any(masks['dense'][frozen[:, 0], frozen[:, 1]])
    assert np.all(offsets['dense'][frozen[:, 0], frozen[:, 1]] == 0.5)
