import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.layers import Input, Add, Average, Concatenate, Dot, Maximum, Minimum, Multiply, Subtract

test_root_path = Path(__file__).parent

merge_layer = [Add, Average, Maximum, Minimum, Multiply, Subtract]
io_type_options = ['io_parallel', 'io_stream']
@pytest.mark.parametrize('merge_layer', merge_layer)
@pytest.mark.parametrize('io_type', io_type_options)
def test_merge(merge_layer, io_type):
    input_shape = (10, 10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = merge_layer()([in1, in2])
    
    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / 'hls4mlprj_merge_{}_{}'.format(merge_layer.__name__.lower(), io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axes', [1])
@pytest.mark.parametrize('io_type', ['io_parallel']) # No io_stream implementation yet
def test_dot(axes, io_type):
    input_shape = (10,) # Only 1D implemented

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Dot(axes=axes)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / 'hls4mlprj_dot_axes_{}_{}'.format(str(axes), io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_concatenate1d(io_type):
    input_shape = (10,)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate()([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / 'hls4mlprj_concatenate1d_{}'.format(io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_concatenate2d(axis, io_type):
    input_shape = (10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate(axis=axis)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path /'hls4mlprj_concatenate2d_axis_{}_{}'.format(str(axis), io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('axis', [1, 2, 3])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_concatenate3d(axis, io_type):
    input_shape = (10, 10, 3)

    in1 = Input(shape=input_shape)
    in2 = Input(shape=input_shape)
    out = Concatenate(axis=axis)([in1, in2])

    model = tf.keras.models.Model(inputs=[in1, in2], outputs=out)
    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path /'hls4mlprj_concatenate3d_axis_{}_{}'.format(str(axis), io_type))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    keras_prediction = model.predict([X_input1, X_input2])
    hls_prediction = hls_model.predict([X_input1, X_input2]).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)
