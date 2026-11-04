from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten

import hls4ml

test_root_path = Path(__file__).parent


def make_model(seed):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.models.Sequential()
    model.add(Dense(8, input_shape=(16,), activation='relu', name='dense1'))
    model.add(Dense(4, name='dense2'))
    return model


def convert(model, output_dir, io_type, reloadable, weight_sets=1):
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<16,6>', backend='Altera'
    )
    for layer in ['dense1', 'dense2']:
        config['LayerName'][layer]['ReloadableWeights'] = reloadable
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend='Altera',
        io_type=io_type,
        weight_sets=weight_sets,
    )
    hls_model.compile()
    return hls_model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('weight_sets', [1, 2])
def test_reloadable_weights(test_case_id, io_type, weight_sets):
    """Loading the weights of another model must give exactly the predictions of that model converted with
    constant weights, and other weight sets must keep their weights."""
    output_dir = test_root_path / test_case_id
    model_a = make_model(1)
    model_b = make_model(2)
    hls_a = convert(model_a, str(output_dir / 'constant_a'), io_type, False)
    hls_b = convert(model_b, str(output_dir / 'constant_b'), io_type, False)
    hls_reloadable = convert(model_a, str(output_dir / 'reloadable'), io_type, True, weight_sets)
    backend = hls4ml.backends.get_backend('Altera')

    X = np.random.default_rng(0).uniform(-2, 2, (20, 16))
    y_a = hls_a.predict(X)
    y_b = hls_b.predict(X)
    assert not np.array_equal(y_a, y_b)

    np.testing.assert_array_equal(hls_reloadable.predict(X), y_a)

    last_set = weight_sets - 1
    backend.load_weights(hls_reloadable, hls_b, weight_set=last_set)
    backend.set_weight_set(hls_reloadable, last_set)
    np.testing.assert_array_equal(hls_reloadable.predict(X), y_b)

    if weight_sets > 1:
        backend.set_weight_set(hls_reloadable, 0)
        np.testing.assert_array_equal(hls_reloadable.predict(X), y_a)


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_constant_weights_unchanged(test_case_id, io_type):
    """Without reloadable weights, none of the reloadable machinery may appear in the generated project."""
    output_dir = test_root_path / test_case_id
    model = make_model(1)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Altera')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(output_dir), backend='Altera', io_type=io_type
    )
    hls_model.write()

    for path in (output_dir / 'src').glob('**/*'):
        if path.is_file() and path.name != 'nnet_reloadable.h':
            text = path.read_text()
            for marker in ['reloadable', 'device_global', 'mode_load_weights', 'WeightLoadPipe']:
                assert marker not in text, f'{marker} found in {path}'
    assert '-std=c++20' not in (output_dir / 'CMakeLists.txt').read_text()


def test_reloadable_unsupported_layer(test_case_id):
    model = tf.keras.models.Sequential()
    model.add(Conv1D(4, 3, input_shape=(16, 2), name='conv'))
    model.add(Flatten())
    model.add(Dense(4, name='dense'))
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Altera')
    config['LayerName']['conv']['ReloadableWeights'] = True
    with pytest.raises(Exception, match='only supported for Dense'):
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=str(test_root_path / test_case_id), backend='Altera'
        )
