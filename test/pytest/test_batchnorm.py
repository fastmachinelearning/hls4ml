from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = 16
atol = 5e-3


@pytest.fixture(scope='module')
def data():
    np.random.seed(0)
    X = np.random.rand(100, in_shape)
    return X


@pytest.fixture(scope='module')
def model(request):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(in_shape,), center=request.param, scale=request.param))
    model.compile()
    return model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize('model', [True, False], indirect=True)
def test_batchnorm(test_case_id, model, data, backend, io_type):
    default_precision = 'fixed<32, 1>'

    config = hls4ml.utils.config_from_keras_model(
        model, default_precision=default_precision, granularity='name', backend=backend
    )
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type=io_type, output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_keras = np.squeeze(model.predict(data))
    y_hls = hls_model.predict(data)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
