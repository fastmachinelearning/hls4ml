from pathlib import Path

import keras
import numpy as np
import pytest

sparsepixels = pytest.importorskip('sparsepixels')

from hgq.config import LayerConfigScope, QuantizerConfigScope  # noqa: E402
from hgq.layers import QDense  # noqa: E402
from hgq.quantizer.config import QuantizerConfig  # noqa: E402
from keras.layers import Flatten  # noqa: E402
from sparsepixels.layers import (  # noqa: E402
    AveragePooling2DSparse,
    InputReduce,
    MaxPooling2DSparse,
    QConv2DSparse,
)

import hls4ml  # noqa: E402

test_root_path = Path(__file__).parent


def _build_sparse_cnn(input_shape=(8, 8, 1), n=4, threshold=0.4, pool='avg'):
    iq_conf = QuantizerConfig(place='datalane', q_type='kif', i0=4, f0=8, overflow_mode='WRAP')
    with (
        QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
        LayerConfigScope(enable_ebops=True, enable_iq=True, beta0=1e-5),
    ):
        x_in = keras.Input(shape=input_shape, name='x_in')
        x, keep_mask = InputReduce(n=n, threshold=threshold, name='input_reduce')(x_in)
        x = QConv2DSparse(
            filters=2,
            kernel_size=3,
            name='conv',
            padding='same',
            strides=1,
            activation='relu',
            iq_conf=iq_conf,
        )([x, keep_mask])
        pool_layer = MaxPooling2DSparse(2, name='pool') if pool == 'max' else AveragePooling2DSparse(2, name='pool')
        x, keep_mask = pool_layer([x, keep_mask])
        x = Flatten(name='flatten')(x)
        x = QDense(1, name='dense', iq_conf=iq_conf)(x)
    return keras.Model(x_in, x, name='cnn_sparse_test')


def _make_sparse_inputs(n_samples, h=8, w=8, n_active_per_sample=4, threshold=0.4):
    x = np.zeros((n_samples, h, w, 1), dtype=np.float32)
    for i in range(n_samples):
        active_idx = np.random.choice(h * w, size=n_active_per_sample, replace=False)
        for idx in active_idx:
            x[i, idx // w, idx % w, 0] = threshold + 0.1 + np.random.rand() * 0.5
    return x


def _convert_and_check(model, x, output_dir, backend, layer_overrides=None):
    y_keras = model.predict(x, verbose=0)

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    for name, overrides in (layer_overrides or {}).items():
        hls_config['LayerName'].setdefault(name, {}).update(overrides)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=str(output_dir),
        backend=backend,
        io_type='io_parallel',
    )
    hls_model.compile()

    # Guard the input-precision regression: bit_exact must propagate the downstream precision
    # request back through the sparse layers to the model input. Otherwise x_in collapses to a
    # degenerate type (e.g. ap_ufixed<1,0>) that clamps the real inputs to {0, 0.5}.
    in_prec = hls_model.graph['x_in'].get_output_variable().type.precision
    assert in_prec.width > 2, f'input precision collapsed to {in_prec}'

    y_hls = hls_model.predict(x).reshape(y_keras.shape)
    mean_abs_diff = float(np.mean(np.abs(y_keras - y_hls)))
    print(f'{output_dir.name}: mean|diff|={mean_abs_diff:.4f}')
    assert mean_abs_diff < 0.05


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('pool', ['avg', 'max'])
def test_sparse_cnn(test_case_id, backend, pool):
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    model = _build_sparse_cnn(pool=pool)
    x = _make_sparse_inputs(n_samples=1000)
    _convert_and_check(model, x, test_root_path / test_case_id, backend)


@pytest.mark.parametrize('backend', ['Vitis'])
def test_sparse_cnn_parallelization(test_case_id, backend):
    # Partial parallelization and the streaming input reduce only change the unroll/implementation,
    # so the numerical output must still match the fully-parallel/tree default.
    np.random.seed(43)
    keras.utils.set_random_seed(43)

    model = _build_sparse_cnn()
    x = _make_sparse_inputs(n_samples=500)
    overrides = {
        'input_reduce': {'Variant': 'stream'},
        'conv': {'PixelParallelFactor': 2, 'FiltParallelFactor': 1},
        'pool': {'PixelParallelFactor': 2, 'ChanParallelFactor': 1},
    }
    _convert_and_check(model, x, test_root_path / test_case_id, backend, layer_overrides=overrides)
