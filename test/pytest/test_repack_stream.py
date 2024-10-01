from pathlib import Path

import numpy as np
import pytest
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult', 'oneAPI'])
def test_repack_precision(backend: str):
    inp = keras.Input(shape=(3, 3), name='inp')
    out = keras.layers.Reshape((3, 3), name='reshape')(inp)
    out = keras.layers.Conv1D(2, 2, name='conv')(out)
    model = keras.Model(inp, out)

    layer_conf = {
        'inp': {'Precision': 'fixed<20,10>'},
        'reshape': {'Precision': 'fixed<20,10>'},
        'conv': {'Precision': 'fixed<20,10>'},
    }

    hls_config = {'Model': {'Precision': 'fixed<2,1>', 'ReuseFactor': 1}, 'LayerName': layer_conf}

    # Repack only happens in io_stream
    model_hls = convert_from_keras_model(
        model,
        backend=backend,
        output_dir=str(test_root_path / f'hls4mlprj_repack_precision_{backend}'),
        hls_config=hls_config,
        io_type='io_stream',
    )
    model_hls.write()  # Not needed for this test, but useful for debugging

    reshape_name = 'reshape' if backend == 'oneAPI' else 'repack_reshape'
    assert reshape_name in model_hls.graph, f'{reshape_name} not found in graph'
    repack_precision = model_hls.graph[reshape_name].attributes['result_t'].precision
    assert repack_precision.integer == 10, 'Precision mismatch'
    assert repack_precision.fractional == 10, 'Precision mismatch'
    assert repack_precision.width == 20, 'Precision mismatch'
    assert repack_precision.signed is True, 'Precision mismatch'


@pytest.mark.parametrize(
    'backend, strategy',
    [
        ('Quartus', 'Resource'),
        ('oneAPI', 'Resource'),
        ('Vivado', 'Resource'),
        ('Vitis', 'Resource'),
        ('Vivado', 'Latency'),
        ('Vitis', 'Latency'),
        ('Catapult', 'Latency'),
        ('Catapult', 'Resource'),
    ],
)
def test_repack(backend: str, strategy: str):
    inp1 = keras.Input(shape=(4,), name='inp1')
    inp2 = keras.Input(shape=(4,), name='inp2')
    r1 = keras.layers.Reshape((2, 2), name='reshape1')(inp1)
    r2 = keras.layers.Reshape((2, 2), name='reshape2')(inp2)
    out = keras.layers.Concatenate(name='concat')([r1, r2])
    model = keras.Model([inp1, inp2], out)

    hls_config = {'Model': {'Precision': 'ap_ufixed<8,8>', 'ReuseFactor': 1}, 'Strategy': strategy}
    model_hls = convert_from_keras_model(
        model,
        io_type='io_stream',
        backend=backend,
        hls_config=hls_config,
        output_dir=str(test_root_path / f'hls4mlprj_repack_{backend}_{strategy}'),
    )
    model_hls.compile()
    inp_data = [
        np.random.randint(0, 2**8, (100, 4)).astype(np.float32),
        np.random.randint(0, 2**8, (100, 4)).astype(np.float32),
    ]
    out_target = np.concatenate([inp_data[0].reshape(100, 2, 2), inp_data[1].reshape(100, 2, 2)], axis=-1)
    out_data: np.ndarray = model_hls.predict(inp_data)  # type: ignore
    assert np.all(out_data.reshape(out_target.shape) == out_target), 'Concatenate failed: mismatching output'
