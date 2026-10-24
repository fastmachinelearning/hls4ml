import numpy as np
import pytest
import tensorflow as tf

import hls4ml


def make_model(dilation, padding, layer_type='Conv1D'):
    inputs = tf.keras.Input(shape=(17, 2))
    outputs = getattr(tf.keras.layers, layer_type)(3, 3, padding=padding, dilation_rate=dilation, name='conv')(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Use fixed weights so the test always produces the same result.
    kernel = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 32 - 0.25
    bias = np.array([-0.125, 0.0, 0.125], dtype=np.float32)
    if layer_type == 'SeparableConv1D':
        depthwise = np.arange(6, dtype=np.float32).reshape(3, 2, 1) / 8 - 0.25
        pointwise = np.arange(6, dtype=np.float32).reshape(1, 2, 3) / 8 - 0.25
        model.get_layer('conv').set_weights([depthwise, pointwise, bias])
    else:
        model.get_layer('conv').set_weights([kernel, bias])
    return model


# Test each dilation rate with both padding modes on Vivado and Vitis.
@pytest.mark.parametrize(
    'layer_type,strategy', [('Conv1D', 'Latency'), ('Conv1D', 'Resource'), ('SeparableConv1D', 'Latency')]
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'dilation,padding',
    [(1, 'valid'), (1, 'same'), (2, 'valid'), (2, 'same'), (4, 'valid'), (4, 'same')],
)
def test_conv1d_dilation(tmp_path, backend, dilation, padding, layer_type, strategy):
    tf.keras.utils.set_random_seed(123)
    model = make_model(dilation, padding, layer_type)
    x = np.arange(2 * 17 * 2, dtype=np.float32).reshape(2, 17, 2) / 64 - 0.5

    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend=backend)
    config['Model']['Precision'] = 'fixed<32,12>'
    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = 1 if strategy == 'Latency' else 3
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend=backend,
        io_type='io_parallel',
        output_dir=str(tmp_path),
    )

    assert tuple(hls_model.get_output_variables()[0].shape) == model.output_shape[1:]

    hls_model.compile()
    y_keras = model(x, training=False).numpy()
    y_hls = hls_model.predict(x).reshape(y_keras.shape)
    np.testing.assert_allclose(y_hls, y_keras, rtol=0, atol=2e-3)


@pytest.mark.parametrize('layer_type', ['Conv1D', 'SeparableConv1D'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_conv1d_dilation_stream_rejected(tmp_path, backend, layer_type):
    model = make_model(2, 'valid', layer_type)
    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend=backend)

    with pytest.raises(NotImplementedError, match='only supported with io_parallel'):
        hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            backend=backend,
            io_type='io_stream',
            output_dir=str(tmp_path),
        )


@pytest.mark.parametrize('layer_type', ['Conv1D', 'SeparableConv1D'])
@pytest.mark.parametrize('padding', ['valid', 'same'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_conv1d_dilation_synthesis(tmp_path, backend, padding, layer_type, synthesis_config):
    if not synthesis_config['run_synthesis']:
        pytest.skip('Set RUN_SYNTHESIS=true to run HLS synthesis')
    model = make_model(2, padding, layer_type)
    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend=backend)
    config['Model']['Precision'] = 'fixed<16,6>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend=backend,
        io_type='io_parallel',
        part='xc7z020clg400-1',
        clock_period=5,
        output_dir=str(tmp_path),
    )
    hls_model.compile()
    report = hls_model.build(csim=False, synth=True, cosim=False, validation=False, export=False)
    assert int(report['CSynthesisReport']['LUT']) > 0
