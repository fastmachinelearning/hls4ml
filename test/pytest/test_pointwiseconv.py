from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D

import hls4ml

test_root_path = Path(__file__).parent

padds_options = ['same', 'valid']
chans_options = ['channels_last']
strides1d_options = [(1,), (2,)]
strides2d_options = [(1, 1), (2, 2)]


@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('strides', strides1d_options)
@pytest.mark.parametrize(
    'backend, io_type, strategy, rf',
    [
        ('Quartus', 'io_parallel', 'resource', 1),
        ('Quartus', 'io_stream', 'resource', 1),
        ('oneAPI', 'io_parallel', 'resource', 1),
        ('oneAPI', 'io_stream', 'resource', 1),
        ('Vivado', 'io_parallel', 'resource', 1),
        ('Vitis', 'io_parallel', 'resource', 1),
        ('Vivado', 'io_parallel', 'latency', 1),
        ('Vitis', 'io_parallel', 'latency', 1),
        ('Vivado', 'io_parallel', 'latency', 14),
        ('Vitis', 'io_parallel', 'latency', 14),
        ('Vivado', 'io_stream', 'latency', 1),
        ('Vivado', 'io_stream', 'resource', 1),
        ('Vitis', 'io_stream', 'latency', 1),
        ('Vitis', 'io_stream', 'resource', 1),
        ('Catapult', 'io_stream', 'latency', 1),
        ('Catapult', 'io_stream', 'resource', 1),
    ],
)
def test_pointwiseconv1d(chans, padds, strides, backend, io_type, strategy, rf):
    model = tf.keras.models.Sequential()
    input_shape = (28, 3)
    model.add(
        Conv1D(
            filters=32,
            kernel_size=(1,),
            strides=strides,
            padding=padds,
            input_shape=input_shape,
            kernel_initializer='normal',
            use_bias=False,
            data_format=chans,
            name='pointwise1d',
        )
    )
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)

    default_precision = 'fixed<32,16>'
    config = hls4ml.utils.config_from_keras_model(model, default_precision=default_precision, granularity='name')
    config['Model']['Strategy'] = strategy
    config['LayerName']['pointwise1d']['ReuseFactor'] = rf

    output_dir = str(
        test_root_path / f'hls4mlprj_pointwise1d_{chans}_{strides[0]}_{padds}_{backend}_{io_type}_{strategy}_rf{rf}'
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    if not (backend in ['Quartus', 'oneAPI'] and io_type == 'io_stream'):
        # Quartus io_stream does not currently have a special pointwise implementation
        assert 'Pointwise' in list(hls_model.graph.values())[1].class_name
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('strides', strides2d_options)
@pytest.mark.parametrize(
    'backend, io_type, strategy',
    [
        ('Quartus', 'io_parallel', 'resource'),
        ('Quartus', 'io_stream', 'resource'),
        ('oneAPI', 'io_parallel', 'resource'),
        ('oneAPI', 'io_stream', 'resource'),
        ('Vivado', 'io_parallel', 'resource'),
        ('Vivado', 'io_parallel', 'latency'),
        ('Vivado', 'io_stream', 'latency'),
        ('Vivado', 'io_stream', 'resource'),
        ('Catapult', 'io_stream', 'latency'),
        ('Catapult', 'io_stream', 'resource'),
    ],
)
def test_pointwiseconv2d(chans, padds, strides, backend, io_type, strategy):
    model = tf.keras.models.Sequential()
    input_shape = (28, 28, 3)
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=strides,
            padding=padds,
            input_shape=input_shape,
            kernel_initializer='normal',
            use_bias=False,
            data_format=chans,
            name='pointwise2d',
        )
    )

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)

    default_precision = 'fixed<32, 9>'

    config = hls4ml.utils.config_from_keras_model(model, default_precision=default_precision)
    config['Model']['Strategy'] = strategy
    stride_cfg = str(strides).replace(', ', '_').replace('(', '').replace(')', '')
    output_dir = str(
        test_root_path / f'hls4mlprj_pointwise2d_{chans}_strides_{stride_cfg}_{padds}_padding_{backend}_{io_type}_{strategy}'
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    if not (backend in ['Quartus', 'oneAPI'] and io_type == 'io_stream'):
        # Quartus io_stream does not currently have a special pointwise implementation
        assert 'Pointwise' in list(hls_model.graph.values())[1].class_name
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('strategy', ['Latency', 'Resource'])
def test_pointwise_config(strategy):
    model = tf.keras.models.Sequential()
    input_shape = (8, 8, 3)
    model.add(
        Conv2D(
            filters=8,
            kernel_size=(1, 1),
            input_shape=input_shape,
            kernel_initializer='normal',
            use_bias=False,
            name='conv2d_1x1',
        )
    )

    model.compile(optimizer='adam', loss='mse')

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    config['Model']['Strategy'] = strategy
    config['LayerName']['conv2d_1x1']['Strategy'] = strategy  # Will fail if the strategy is not lowercase
    output_dir = str(test_root_path / f'hls4mlprj_pointwise2d_config_{strategy}')

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)
    # Model will fail to compile if strategy was set incorrectly
    hls_model.compile()
