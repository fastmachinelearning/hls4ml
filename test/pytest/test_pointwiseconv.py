import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras import backend as K

test_root_path = Path(__file__).parent

padds_options = ['same', 'valid']
chans_options = ['channels_last']
io_type_options = ['io_parallel', 'io_stream']
strides1d_options = [(1,), (2,)]
strides2d_options = [(1, 1), (2, 2)]
strategy_options = ['Latency', 'Resource']

@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("strides", strides1d_options)
@pytest.mark.parametrize("io_type", io_type_options)
@pytest.mark.parametrize("strategy", strategy_options)
def test_pointwiseconv1d(chans, padds, strides, io_type, strategy):
    model = tf.keras.models.Sequential()
    input_shape = (28, 3)
    model.add(Conv1D(filters=32,
                     kernel_size=(1,),
                     strides=strides,
                     padding=padds,
                     input_shape=input_shape,
                     kernel_initializer='normal',
                     use_bias=False,
                     data_format=chans
                     ))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / 'hls4mlprj_pointwise1d_{}_strides_{}_{}_padding_{}_{}'.format(chans, strides[0], padds, io_type, strategy))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    assert 'Pointwise' in list(hls_model.graph.values())[1].class_name
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)

@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("strides", strides2d_options)
@pytest.mark.parametrize("io_type", io_type_options)
@pytest.mark.parametrize("strategy", strategy_options)
def test_pointwiseconv2d(chans, padds, strides, io_type, strategy):
    model = tf.keras.models.Sequential()
    input_shape = (28, 28, 3)
    model.add(Conv2D(filters=32,
                     kernel_size=(1, 1),
                     strides=strides,
                     padding=padds,
                     input_shape=input_shape,
                     kernel_initializer='normal',
                     use_bias=False,
                     data_format=chans
                     ))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    config['Model']['Strategy'] = strategy
    stride_cfg = str(strides).replace(', ', '_').replace('(', '').replace(')', '')
    output_dir = str(test_root_path / 'hls4mlprj_pointwise2d_{}_strides_{}_{}_padding_{}_{}'.format(chans, stride_cfg, padds, io_type, strategy))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    assert 'Pointwise' in list(hls_model.graph.values())[1].class_name
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)