from pathlib import Path

import numpy as np
import pytest
from qkeras.qlayers import QActivation
from qkeras.quantizers import quantized_relu
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent


def randX(batch_size, N):
    return np.random.rand(batch_size, N)


@pytest.fixture(scope='module')
def randX_1000_1():
    return randX(1000, 1)


@pytest.mark.parametrize(
    'quantizer',
    [
        (quantized_relu(4, negative_slope=0.5)),
        (quantized_relu(4, 2, negative_slope=0.5)),
        (quantized_relu(8, negative_slope=0.125)),
        (quantized_relu(8, 4, negative_slope=1.0)),
        (quantized_relu(10, negative_slope=0.25)),
        (quantized_relu(10, 5, negative_slope=0.5)),
        (quantized_relu(10, 5, negative_slope=0.25)),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantizer(randX_1000_1, quantizer, backend, io_type):
    '''
    Test a single quantizer as an Activation function.
    Checks the type inference through the conversion is correct without just
    using the same logic.
    '''
    X = randX_1000_1
    X = np.round(X * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>
    model = Sequential()
    model.add(QActivation(input_shape=(1,), activation=quantizer, name='quantizer'))
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    output_dir = str(
        test_root_path
        / 'hls4mlprj_qkeras_quantizer_{}_{}_{}_{}_{}'.format(
            quantizer.__class__.__name__, quantizer.bits, quantizer.integer, backend, io_type
        )
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    y_qkeras = model.predict(X)
    y_hls4ml = hls_model.predict(X)
    np.testing.assert_allclose(y_hls4ml, y_qkeras, rtol=1e-5, atol=0)
