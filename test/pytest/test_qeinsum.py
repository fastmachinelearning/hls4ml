from pathlib import Path

import keras
import numpy as np
import pytest

from hls4ml.converters import convert_from_keras_model

try:
    from hgq.layers import QEinsum
    from hgq.utils import trace_minmax
except ImportError:
    pytest.skip('HGQ2 is not installed', allow_module_level=True)

from keras.layers import Input

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'operation',
    [
        # eq, inp, out
        ('xbi,xj->xbij', (8, 16), (16,)),
        ('xbi,xio->xbo', (7, 8), (8, 9)),
        ('xi,xoi->xo', (16,), (20, 16)),
        ('xabcd,xbcde->xaeb', (2, 4, 8, 16), (4, 8, 16, 3)),
    ],
)
def test_einsum_dense(backend, io_type, strategy, operation):
    eq, inp0_shape, inp1_shape = operation
    inp0 = Input(inp0_shape)
    inp1 = Input(inp1_shape)
    out = QEinsum(eq, name='einsum')([inp0, inp1])
    model = keras.Model(inputs=[inp0, inp1], outputs=out)

    data = np.random.randn(1000, *inp0_shape).astype(np.float32), np.random.randn(1000, *inp1_shape).astype(np.float32)
    eq_name = eq.replace(',', '_').replace('->', '_')
    output_dir = str(test_root_path / f'hls4mlprj_einsum_{eq_name}_{backend}_{io_type}_{strategy}')
    hls_config = {'Model': {'Precision': 'ap_fixed<1,0>', 'ReuseFactor': 1}, 'Strategy': strategy}

    r_keras = trace_minmax(model, data, batch_size=8192, verbose=0, return_results=True)  # type: ignore

    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)  # type: ignore

    assert np.all(r_hls.ravel() == r_keras.ravel())
