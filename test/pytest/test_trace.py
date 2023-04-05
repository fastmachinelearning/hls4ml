from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense

import hls4ml
import hls4ml.model.profiling

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_trace(backend):
    '''Test the tracing feature with a simple Keras model.'''
    model = tf.keras.models.Sequential()
    model.add(
        Dense(
            2,
            input_shape=(1,),
            name='Dense',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
    )
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, 1)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True

    output_dir = str(test_root_path / f'hls4mlprj_trace_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend)

    hls_model.compile()
    hls4ml_pred, hls4ml_trace = hls_model.trace(X_input)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_input)

    np.testing.assert_allclose(hls4ml_trace['Dense'], keras_trace['Dense'], rtol=1e-2, atol=0.01)
    np.testing.assert_allclose(hls4ml_pred, keras_prediction, rtol=1e-2, atol=0.01)
