from pathlib import Path

import keras
import numpy as np
import pytest

sparsepixels = pytest.importorskip('sparsepixels')

from hgq.config import LayerConfigScope, QuantizerConfigScope  # noqa: E402
from hgq.layers import QDense  # noqa: E402
from hgq.quantizer.config import QuantizerConfig  # noqa: E402
from keras.layers import Flatten  # noqa: E402
from sparsepixels.layers import AveragePooling2DSparse, InputReduce, QConv2DSparse  # noqa: E402

import hls4ml  # noqa: E402

test_root_path = Path(__file__).parent


def _build_sparse_cnn(input_shape=(8, 8, 1), n_max_pixels=4, threshold=0.4):
    iq_conf = QuantizerConfig(place='datalane', q_type='kif', i0=4, f0=8, overflow_mode='WRAP')
    with (
        QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
        LayerConfigScope(enable_ebops=True, enable_iq=True, beta0=1e-5),
    ):
        x_in = keras.Input(shape=input_shape, name='x_in')
        x, keep_mask = InputReduce(n_max_pixels=n_max_pixels, threshold=threshold, name='input_reduce')(x_in)
        x = QConv2DSparse(
            filters=2,
            kernel_size=3,
            name='conv',
            padding='same',
            strides=1,
            activation='relu',
            iq_conf=iq_conf,
        )([x, keep_mask])
        x, keep_mask = AveragePooling2DSparse(2, name='pool')([x, keep_mask])
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


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_sparse_cnn(test_case_id, backend):
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    model = _build_sparse_cnn()
    x = _make_sparse_inputs(n_samples=1000)

    y_keras = model.predict(x, verbose=0)

    output_dir = test_root_path / test_case_id
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=str(output_dir),
        backend=backend,
        io_type='io_parallel',
    )
    hls_model.compile()

    y_hls = hls_model.predict(x).reshape(y_keras.shape)

    mean_abs_diff = float(np.mean(np.abs(y_keras - y_hls)))
    print(f'sparse-pixels {backend}: mean|diff|={mean_abs_diff:.4f}')

    assert mean_abs_diff < 0.5
