import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import optimizers
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras import backend as K

test_root_path = Path(__file__).parent

keras_conv2d = [SeparableConv2D]
padds_options = ['same', 'valid']
chans_options = ['channels_last']
io_type_options = ['io_stream']
strides_options = [(1, 1), (2, 2)]
bias_options = [False]
@pytest.mark.parametrize("conv2d", keras_conv2d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("strides", strides_options)
@pytest.mark.parametrize("bias", bias_options)
@pytest.mark.parametrize("io_type", io_type_options)
def test_sepconv2d(conv2d, chans, padds, strides, bias, io_type):
    model = tf.keras.models.Sequential()
    input_shape = (28, 28, 3)
    model.add(conv2d(filters=32,
                     kernel_size=(2, 2),
                     strides=strides,
                     padding=padds,
                     input_shape=input_shape,
                     kernel_initializer='normal',
                     use_bias=bias,
                     data_format=chans
                     ))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / 'hls4mlprj_{}_{}_{}'.format(conv2d.__name__.lower(), chans, padds))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, io_type=io_type)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.01)

