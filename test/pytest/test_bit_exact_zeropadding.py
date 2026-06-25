"""Bit-exact precision propagation through ZeroPadding1D/2D.

ZeroPadding inserts exact zeros, so the bit-exact flow must propagate the
input kif unchanged (padded with zeros) through the layer. Placing a plain
Keras ZeroPadding layer between two HGQ2 quantized convolutions triggers the
bit_exact pass and exercises the ZeroPadding _produce_kif handlers.
"""

from pathlib import Path

import keras
import numpy as np
import pytest

from hls4ml.converters import convert_from_keras_model

try:
    from hgq.config import QuantizerConfigScope
    from hgq.layers import QConv1D, QConv2D
    from hgq.utils import trace_minmax
except ImportError:
    pytest.skip('HGQ2 is not installed', allow_module_level=True)

from keras.layers import Input, ZeroPadding1D, ZeroPadding2D  # noqa: E402

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_bit_exact_zeropadding1d(test_case_id, backend, io_type):
    """ZeroPadding1D between two quantized Conv1D layers must convert via the
    bit_exact flow and reproduce the quantized Keras output exactly."""
    with QuantizerConfigScope(f0=4, i0=4):
        inp = Input((16, 8))
        x = QConv1D(4, 1, name='c0')(inp)
        x = ZeroPadding1D(padding=(2, 0))(x)
        out = QConv1D(4, 3, padding='valid', name='c1')(x)
        model = keras.Model(inp, out)

    data = np.random.default_rng(0).standard_normal((1000, 16, 8)).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    precision = 'ac_fixed<2,0>' if backend == 'oneAPI' else 'ap_fixed<1,0>'
    hls_config = {'Model': {'Precision': precision, 'ReuseFactor': 1, 'Strategy': 'latency'}}
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )
    hls_model.compile()

    r_hls = hls_model.predict(data).reshape(r_keras.shape)
    np.testing.assert_array_equal(r_keras, r_hls)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_bit_exact_zeropadding2d(test_case_id, backend, io_type):
    """ZeroPadding2D between two quantized Conv2D layers must convert via the
    bit_exact flow and reproduce the quantized Keras output exactly."""
    with QuantizerConfigScope(f0=4, i0=4):
        inp = Input((8, 8, 4))
        x = QConv2D(8, 1, name='c0')(inp)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        out = QConv2D(8, 3, padding='valid', name='c1')(x)
        model = keras.Model(inp, out)

    data = np.random.default_rng(1).standard_normal((500, 8, 8, 4)).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    precision = 'ac_fixed<2,0>' if backend == 'oneAPI' else 'ap_fixed<1,0>'
    hls_config = {'Model': {'Precision': precision, 'ReuseFactor': 1, 'Strategy': 'latency'}}
    output_dir = str(test_root_path / test_case_id)
    hls_model = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )
    hls_model.compile()

    r_hls = hls_model.predict(data).reshape(r_keras.shape)
    np.testing.assert_array_equal(r_keras, r_hls)
