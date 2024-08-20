from glob import glob
from pathlib import Path

import keras
import numpy as np
import pytest

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('i', [4, 8, 10])
@pytest.mark.parametrize('f', [-2, 0, 2, 7, 14])
def test_weight_writer(k, i, f):
    k, b, i = k, k + i + f, k + i
    w = np.array([[np.float32(2.0**-f)]])
    u = '' if k else 'u'
    dtype = f'{u}fixed<{b}, {i}>'
    hls_config = {'LayerName': {'dense': {'Precision': {'weight': dtype}}}}

    model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,), name='dense')])
    model.layers[0].kernel.assign(keras.backend.constant(w))
    output_dir = str(test_root_path / f'hls4ml_prj_test_weight_writer_{dtype}')

    model_hls = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_config, output_dir=output_dir, write_weights_txt=True
    )
    model_hls.write()

    w_paths = glob(str(Path(output_dir) / 'firmware/weights/w*.txt'))
    assert len(w_paths) == 1

    w_loaded = np.loadtxt(w_paths[0], delimiter=',').reshape(1, 1)
    assert np.all(w == w_loaded)
