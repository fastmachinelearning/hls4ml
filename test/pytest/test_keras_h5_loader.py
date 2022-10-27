import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path


test_root_path = Path(__file__).parent

test_root_path = Path('/tmp')


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_keras_h5_loader(backend):
    input_shape = (10,)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Activation(activation='relu'),
    ])

    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    config = {'OutputDir': f'KerasH5_loader_test_{backend}',
              'ProjectName': f'KerasH5_loader_test_{backend}',
              'Backend': backend,
              'ClockPeriod': 25.0,
              'IOType': 'io_parallel',
              'HLSConfig': hls_config,
              'KerasH5': str(test_root_path / f'KerasH5_loader_test_{backend}.h5'),
              'output_dir': str(test_root_path / f'KerasH5_loader_test_{backend}')}

    model.save(config['KerasH5'])
    hls_model = hls4ml.converters.keras_to_hls(config)
    hls_model.compile()
    data = np.random.rand(1000, 10).astype(np.float32)
    pred = hls_model.predict(data)
    np.testing.assert_allclose(pred, model.predict(data), rtol=5e-3, atol=5e-3)
