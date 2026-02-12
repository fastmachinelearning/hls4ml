from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from conftest import get_pytest_case_id

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
def test_keras_h5_loader(request, backend):
    input_shape = (10,)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Activation(activation='relu'),
        ]
    )

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    output_dir = str(test_root_path / get_pytest_case_id(request))
    config = {
        'OutputDir': output_dir,
        'ProjectName': f'KerasH5_loader_test_{backend}',
        'Backend': backend,
        'ClockPeriod': 25.0,
        'IOType': 'io_parallel',
        'HLSConfig': hls_config,
        'KerasH5': str(Path(output_dir) / 'model.h5'),
    }

    model.save(config['KerasH5'])
    hls_model = hls4ml.converters.keras_v2_to_hls(config)
    hls_model.compile()
    data = np.random.rand(1000, 10).astype(np.float32)
    pred = hls_model.predict(data)
    np.testing.assert_allclose(pred, model.predict(data), rtol=5e-3, atol=5e-3)
