import numpy as np
import pytest
import tensorflow as tf

import hls4ml


def make_model(dilation, padding):
    inputs = tf.keras.Input(shape=(17, 2))
    outputs = tf.keras.layers.Conv1D(3, 3, padding=padding, dilation_rate=dilation, name='conv')(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Use fixed weights so the test always produces the same result.
    kernel = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 32 - 0.25
    bias = np.array([-0.125, 0.0, 0.125], dtype=np.float32)
    model.get_layer('conv').set_weights([kernel, bias])
    return model


# Test each dilation rate with both padding modes on Vivado and Vitis.
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'dilation,padding',
    [(1, 'valid'), (1, 'same'), (2, 'valid'), (2, 'same'), (4, 'valid'), (4, 'same')],
)
def test_conv1d_dilation(tmp_path, backend, dilation, padding):
    tf.keras.utils.set_random_seed(123)
    model = make_model(dilation, padding)
    x = np.arange(2 * 17 * 2, dtype=np.float32).reshape(2, 17, 2) / 64 - 0.5

    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend=backend)
    config['Model']['Precision'] = 'fixed<32,12>'
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_conv1d_dilation_stream_rejected(tmp_path, backend):
    model = make_model(2, 'valid')
    config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend=backend)

    with pytest.raises(NotImplementedError, match='only supported with io_parallel'):
        hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            backend=backend,
            io_type='io_stream',
            output_dir=str(tmp_path),
        )
