from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Dense,
)

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_dense(backend, io_type):
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

    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_dense_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    data = hls_model.build()
    print(data)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0]._name
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
    assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
