import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning.attributes import get_attributes_from_keras_model
from hls4ml.optimization.dsp_aware_pruning.objectives import ParameterEstimator


# Test attempts to verify one of the estimators (parameter) is correctly declared, the functions are static etc.
def test_parameter_objective():
    # Model parameters
    dense_units = 16
    conv_filters = 6
    conv_channels = 3
    conv_shape = (3, 3)
    input_shape = (8, 8)

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
    model.add(Flatten(name='flatten'))
    model.add(Dense(dense_units, name='dense', kernel_initializer='ones'))
    model_attributes = get_attributes_from_keras_model(model)

    # Identify optimizable layers and the suitable structure
    for layer in model.layers:
        optimizable, optimization_attributes = ParameterEstimator.is_layer_optimizable(model_attributes[layer.name])
        model_attributes[layer.name].optimizable = optimizable
        model_attributes[layer.name].optimization_attributes = optimization_attributes

    # Verify conv2d and dense are optimizable, flatten is not
    assert model_attributes['conv2d'].optimizable
    assert not model_attributes['flatten'].optimizable
    assert model_attributes['dense'].optimizable

    # Verify layer resources (number of parameters) are correct
    assert [conv_filters * conv_channels * np.prod(conv_shape)] == ParameterEstimator.layer_resources(
        model_attributes['conv2d']
    )
    assert [0] == ParameterEstimator.layer_resources(model_attributes['flatten'])
    assert [conv_filters * np.prod(input_shape) * dense_units] == ParameterEstimator.layer_resources(
        model_attributes['dense']
    )

    # Verify layer savings are correct - is_layer_optimizable should have returned UNSTRUCTURED as the pruning type
    # Since it wasn't overwritten, each pruning step saves one parameter
    assert [1] == ParameterEstimator.layer_savings(model_attributes['conv2d'])
    assert [0] == ParameterEstimator.layer_savings(model_attributes['flatten'])
    assert [1] == ParameterEstimator.layer_savings(model_attributes['dense'])
