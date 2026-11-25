from pathlib import Path

import keras
import pytest

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)
import numpy as np
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QSoftmax
from hgq.utils import trace_minmax

from hls4ml.converters import convert_from_keras_model

test_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vitis', 'Altera'])
def test_hgq2_softmax_large_tables(test_case_id, backend):
    """Bit-exact tables larger than TableSize must be kept at full size to stay bit-exact."""
    with QuantizerConfigScope(f0=9, i0=4):
        conf = QuantizerConfig('kif', 'datalane', f0=9, i0=4)
        inp = keras.layers.Input((8,))
        out = QSoftmax(stable=True, exp_iq_conf=conf, inv_iq_conf=conf, enable_iq=True, enable_oq=True, name='smax')(inp)
        model = keras.Model(inp, out)

    data = (np.random.default_rng(0).standard_normal((1000, 8)) * 3).astype(np.float32)
    r_keras = trace_minmax(model, data, return_results=True)

    with pytest.warns(UserWarning, match='exceed TableSize'):
        model_hls = convert_from_keras_model(
            model,
            output_dir=str(test_path / test_case_id),
            backend=backend,
            io_type='io_parallel',
            hls_config={'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}},
        )
    assert model_hls.graph['smax'].get_attr('exp_table_size') > 1024

    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)
    np.testing.assert_array_equal(r_hls, r_keras)
