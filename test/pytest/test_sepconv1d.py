from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv1D

import hls4ml

test_root_path = Path(__file__).parent

keras_conv1d = [SeparableConv1D]
padds_options = ['same', 'valid']
chans_options = ['channels_last']
io_type_options = ['io_stream']
strides_options = [(1), (2)]
kernel_options = [(1), (3)]
bias_options = [False]


@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("strides", strides_options)
@pytest.mark.parametrize("kernels", kernel_options)
@pytest.mark.parametrize("bias", bias_options)
@pytest.mark.parametrize("io_type", io_type_options)
@pytest.mark.parametrize('backend', ['Vitis'])
def test_sepconv1d(conv1d, chans, padds, strides, kernels, bias, io_type, backend):
    model = tf.keras.models.Sequential()
    input_shape = (28, 3)
    model.add(
        conv1d(
            filters=32,
            kernel_size=kernels,
            strides=strides,
            padding=padds,
            input_shape=input_shape,
            kernel_initializer='normal',
            use_bias=bias,
            data_format=chans,
        )
    )

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    stride_cfg = str(strides).replace(', ', '_').replace('(', '').replace(')', '')
    kernel_cfg = str(kernels).replace(', ', '_').replace('(', '').replace(')', '')
    output_dir = str(
        test_root_path
        / 'hls4mlprj_{}_{}_strides_{}_kernels_{}_{}_padding_{}_{}'.format(
            conv1d.__name__.lower(), chans, stride_cfg, kernel_cfg, padds, backend, io_type
        )
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)


@pytest.mark.parametrize('backend', ['Vivado'])
def test_sepconv1d_build(backend):
    model = tf.keras.models.Sequential()
    input_shape = (28, 3)
    model.add(
        SeparableConv1D(
            filters=16,
            kernel_size=3,
            input_shape=input_shape,
        )
    )
    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Latency'

    cfg = hls4ml.converters.create_config(backend=backend)
    cfg['IOType'] = 'io_stream'
    cfg['HLSConfig'] = config
    cfg['KerasModel'] = model
    cfg['OutputDir'] = 'hls4ml_prj'
    cfg['Part'] = 'xcku115-flvb2104-2-i'

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    hls_model.build(reset=True, csim=False, synth=True)
