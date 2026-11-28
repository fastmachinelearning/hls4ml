from pathlib import Path

import keras
import pytest

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)
import numpy as np
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QDense
from hgq.quantizer import Quantizer
from hgq.utils import trace_minmax

from hls4ml.converters import convert_from_keras_model

test_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vitis', 'Altera'])
@pytest.mark.parametrize('heterogeneous', [False, True])
@pytest.mark.parametrize('overflow_mode', ['SAT', 'WRAP'])
@pytest.mark.parametrize('signed', [False, True])
def test_hgq2_1bit_activation(test_case_id, signed, overflow_mode, heterogeneous, backend):
    """1-bit activation quantizers must round and saturate/wrap exactly, also where ac_fixed needs two bits."""
    keras.utils.set_random_seed(0)
    k = int(signed)
    conf = QuantizerConfig(
        'kif', 'datalane', k0=k, i0=1 - k, f0=0, round_mode='RND', overflow_mode=overflow_mode, trainable=False
    )
    inp = keras.layers.Input((8,))
    with QuantizerConfigScope(f0=4, i0=3):
        x = QDense(8, name='dense')(inp)
    quantizer = Quantizer(conf, name='q')
    model = keras.Model(inp, quantizer(x))
    if heterogeneous:
        # Every other channel gets three fractional bits, the rest stay 1-bit
        f = next(w for w in quantizer.weights if w.path.endswith('/f'))
        f.assign(np.array([0, 3] * 4, dtype=np.float32).reshape(f.shape))

    data = np.random.default_rng(0).uniform(-3, 3, (1000, 8)).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    model_hls = convert_from_keras_model(
        model,
        output_dir=str(test_path / test_case_id),
        backend=backend,
        io_type='io_parallel',
        hls_config={'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}},
    )
    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)
    np.testing.assert_array_equal(r_hls, r_keras)
