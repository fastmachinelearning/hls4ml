from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dense, Input

import hls4ml
import hls4ml.model.profiling

test_root_path = Path(__file__).parent


@pytest.mark.parametrize(
    'backend, io_type',
    [
        ('Vivado', 'io_parallel'),
        ('Vitis', 'io_parallel'),
        ('Quartus', 'io_parallel'),
        ('Altera', 'io_parallel'),
        ('Altera', 'io_stream'),
    ],
)
@pytest.mark.parametrize('activation', ['relu', None])
def test_trace(test_case_id, backend, io_type, activation):
    """Test the tracing feature with a simple Keras model."""
    inputs = Input(shape=(1,))
    x = Dense(
        2,
        name='Dense',
        activation=activation,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )(inputs)
    outputs = Activation(activation='elu', name='Activation')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, 1)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True

    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()
    hls4ml_pred, hls4ml_trace = hls_model.trace(X_input)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_input)
    assert keras_trace.keys() == hls4ml_trace.keys()
    for key in hls4ml_trace.keys():
        np.testing.assert_allclose(hls4ml_trace[key], keras_trace[key], rtol=1e-2, atol=0.01)
    np.testing.assert_allclose(hls4ml_pred, keras_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Altera'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_trace_conv1d(test_case_id, backend, io_type):
    """Test tracing of multi-dimensional layer outputs, which are split over several transfers in io_stream."""
    inputs = Input(shape=(8, 2))
    x = Conv1D(4, 3, name='Conv1D', activation='relu')(inputs)
    outputs = Activation(activation='elu', name='Activation')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(20, 8, 2)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True

    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()
    hls4ml_pred, hls4ml_trace = hls_model.trace(X_input)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_input)
    assert keras_trace.keys() == hls4ml_trace.keys()
    for key in hls4ml_trace.keys():
        np.testing.assert_allclose(hls4ml_trace[key], keras_trace[key], rtol=1e-2, atol=0.01)
    np.testing.assert_allclose(hls4ml_pred.reshape(keras_prediction.shape), keras_prediction, rtol=1e-2, atol=0.01)
