from pathlib import Path

import keras
import pytest

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)

import numpy as np
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hgq.utils import trace_minmax

from hls4ml.converters import convert_from_keras_model

test_path = Path(__file__).parent


@pytest.mark.parametrize('strategy', ('latency', 'distributed_arithmetic'))
def test_hgq2_mha(strategy):
    with QuantizerConfigScope(f0=3, i0=2):
        q = keras.layers.Input((8, 9))
        k = keras.layers.Input((12, 7))
        v = keras.layers.Input((12, 7))
        out = QMultiHeadAttention(4, 8, name='mha', fuse='none')(q, k, v)
        model = keras.Model([q, v, k], out)

    data_q = np.random.randn(10000, 8, 9).astype(np.float32) * 3
    data_v = np.random.randn(10000, 12, 7).astype(np.float32) * 3
    data_k = np.random.randn(10000, 12, 7).astype(np.float32) * 3
    data = [data_q, data_v, data_k]

    r_keras = trace_minmax(model, data, return_results=True)

    model_hls = convert_from_keras_model(
        model,
        output_dir=str(test_path / f'test_hgq2_mha_{strategy}'),
        io_type='io_parallel',
        hls_config={'Model': {'Precision': 'ap_fixed<1,0>', 'ReuseFactor': 1, 'Strategy': strategy}},
    )

    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)  # type: ignore

    mismatches = np.where(r_hls != r_keras)[0]
    assert len(mismatches) == 0, f'Found {len(mismatches)} mismatches'
    assert np.std(r_hls.ravel()) > 0.5, 'Standard deviation is too low'
