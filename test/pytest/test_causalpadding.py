from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

atol = 5e-3


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_causalpadding(io_type, backend):
    model = Sequential()
    model.add(Conv1D(1, 5, padding="causal", input_shape=(100, 1)))
    model.compile()

    data = np.random.randint(0, 10, 100).astype(float)
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=-1)

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision='ap_fixed<32,16>', granularity='name', backend=backend
    )
    odir = str(test_root_path / f'hls4mlprj_validpadding_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
