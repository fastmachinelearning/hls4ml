from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Add, Average, Concatenate, Dot, Input, Maximum, Minimum, Multiply, Subtract

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('merge_layer', [Add, Average, Maximum, Minimum, Multiply, Subtract])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('swap_inputs', [True, False])
def test_merge(merge_layer, io_type, backend, swap_inputs):
    input_shape = (10, 10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    if swap_inputs:
        out = merge_layer()([in2, in1])
    else:
        out = merge_layer()([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(
        test_root_path
        / f'hls4mlprj_merge_{"swap_inputs_" if swap_inputs else ""}{merge_layer.__name__.lower()}_{backend}_{io_type}'
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axes', [1])
@pytest.mark.parametrize('io_type', ['io_parallel'])  # No io_stream implementation yet
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_dot(axes, io_type, backend):
    # Only 1D implemented
    input_shape = (10,)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Dot(axes=axes)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / f'hls4mlprj_dot_axes_{str(axes)}_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_concatenate1d(io_type, backend):
    input_shape = (10,)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate()([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / f'hls4mlprj_concatenate1d_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_concatenate2d(axis, io_type, backend):
    input_shape = (10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate(axis=axis)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / f'hls4mlprj_concatenate2d_axis_{str(axis)}_{io_type}_{backend}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axis', [1, 2, 3])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_concatenate3d(axis, io_type, backend):
    input_shape = (10, 10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate(axis=axis)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / f'hls4mlprj_concatenate3d_axis_{str(axis)}_{io_type}_{backend}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)
