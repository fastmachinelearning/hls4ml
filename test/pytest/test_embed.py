from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = np.random.randint(10, size=(32, 100))
    return X


@pytest.fixture(scope='module')
def keras_model():
    inputs = Input(shape=(100,), name='embedding_input')
    embedding = Embedding(13, 8, input_length=100, name='embedding')(inputs)
    model = Model(inputs=inputs, outputs=embedding)
    return model


@pytest.fixture
def hls_model(keras_model, request):
    backend, io_type = request.param
    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model, default_precision='ap_fixed<16,6>', granularity='name', backend=backend
    )
    hls_config['LayerName']['embedding_input']['Precision']['result'] = 'ap_uint<4>'
    out_dir = str(test_root_path / 'hls4mlprj_embed_{}_{}').format(backend, io_type)
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, backend=backend, hls_config=hls_config, io_type=io_type, output_dir=out_dir
    )

    hls_model.compile()
    return hls_model


@pytest.mark.parametrize(
    'hls_model',
    [
        ('Vivado', 'io_parallel'),
        ('Vitis', 'io_parallel'),
        ('Quartus', 'io_parallel'),
        ('Catapult', 'io_parallel'),
        ('oneAPI', 'io_parallel'),
        ('Vivado', 'io_stream'),
        ('Vitis', 'io_stream'),
        ('Quartus', 'io_stream'),
        ('Catapult', 'io_stream'),
        ('oneAPI', 'io_stream'),
    ],
    ids=[
        'vivado_parallel',
        'vitis_parallel',
        'quartus_parallel',
        'catapult_parallel',
        'oneapi_parallel',
        'vivado_stream',
        'vitis_stream',
        'quartus_stream',
        'catapult_stream',
        'oneapi_stream',
    ],
    indirect=True,
)
def test_embedding_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model
    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X.astype(float)).reshape(y_keras.shape)
    # "accuracy" of hls4ml predictions vs keras
    np.testing.assert_allclose(y_keras, y_hls4ml, rtol=0, atol=1e-03, verbose=True)
