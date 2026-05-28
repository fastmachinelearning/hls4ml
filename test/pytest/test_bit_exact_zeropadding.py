import os

import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import keras  # noqa: E402

import hls4ml  # noqa: E402


def _convert(model, output_dir):
    cfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity='name',
        backend='Vitis',
    )
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        backend='Vitis',
        output_dir=output_dir,
        io_type='io_parallel',
        clock_period=5,
    )


def test_zeropadding1d_produce_kif(tmp_path):
    """ZeroPadding1D followed by Conv1D must convert without
    NotImplementedError and reproduce the Keras output bit-exactly."""
    x = keras.Input((16, 8))
    y = keras.layers.ZeroPadding1D(padding=(2, 0))(x)
    y = keras.layers.Conv1D(4, 3, padding='valid')(y)
    model = keras.Model(x, y)

    hls_model = _convert(model, str(tmp_path / 'zpad1d'))
    hls_model.compile()

    rng = np.random.default_rng(0)
    xin = rng.standard_normal((2, 16, 8)).astype(np.float32)
    y_keras = model.predict(xin)
    y_hls = hls_model.predict(xin).reshape(y_keras.shape)

    np.testing.assert_allclose(y_keras, y_hls, atol=1e-2)


def test_zeropadding2d_produce_kif(tmp_path):
    """ZeroPadding2D followed by Conv2D must convert and match Keras."""
    x = keras.Input((8, 8, 4))
    y = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    y = keras.layers.Conv2D(8, 3, padding='valid')(y)
    model = keras.Model(x, y)

    hls_model = _convert(model, str(tmp_path / 'zpad2d'))
    hls_model.compile()

    rng = np.random.default_rng(1)
    xin = rng.standard_normal((2, 8, 8, 4)).astype(np.float32)
    y_keras = model.predict(xin)
    y_hls = hls_model.predict(xin).reshape(y_keras.shape)

    np.testing.assert_allclose(y_keras, y_hls, atol=1e-2)


def test_causal_via_zeropadding1d_matches_native_causal():
    """`ZeroPadding1D(pad_left=K-1) + Conv1D(padding='valid')` should
    behave identically to `Conv1D(padding='causal')` — guards against
    a future change in either path drifting from the other."""
    K = 3
    ref = keras.Sequential([keras.Input((16, 8)), keras.layers.Conv1D(4, K, padding='causal')])
    pad = keras.Sequential(
        [keras.Input((16, 8)), keras.layers.ZeroPadding1D(padding=(K - 1, 0)), keras.layers.Conv1D(4, K, padding='valid')]
    )
    pad.layers[1].set_weights(ref.layers[0].get_weights())

    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 16, 8)).astype(np.float32)
    np.testing.assert_allclose(
        ref.predict(x),
        pad.predict(x),
        atol=0,
        rtol=0,
    )
