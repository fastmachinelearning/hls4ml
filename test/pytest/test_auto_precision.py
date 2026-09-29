from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Flatten,
    ReLU,
    SeparableConv1D,
    SeparableConv2D,
)
from tensorflow.keras.models import Sequential

import hls4ml
from hls4ml.model.optimizer.passes.infer_precision import _get_precision_from_constant

test_root_path = Path(__file__).parent


# XLS tests are slow due to big IR size, we reduce dimensions to make it faster.
def in_height(backend):
    if backend == 'XLS':
        return 8
    return 10


def in_width(backend):
    if backend == 'XLS':
        return 8
    return 12


def in_feat(backend):
    if backend == 'XLS':
        return 2
    return 4


def input_shape_1d(backend):
    return (in_feat(backend),)


def input_shape_2d(backend):
    return in_width(backend), in_feat(backend)


def input_shape_3d(backend):
    return in_height(backend), in_width(backend), in_feat(backend)


@pytest.fixture()
def data_1d(backend):
    X = np.random.rand(100, *input_shape_1d(backend))
    return X


@pytest.fixture()
def data_2d(backend):
    X = np.random.rand(100, *input_shape_2d(backend))
    return X


@pytest.fixture()
def data_3d(backend):
    X = np.random.rand(100, *input_shape_3d(backend))
    return X


@pytest.fixture()
def keras_model_dense(backend):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=input_shape_1d(backend), name='first_layer'))
    model.add(BatchNormalization(name='first_bn'))
    model.add(Dense(6, activation='relu', name='middle_layer'))
    model.add(BatchNormalization(name='middle_bn'))
    model.add(Dense(4, activation='relu', name='last_layer'))
    model.compile()
    return model


@pytest.fixture()
def keras_model_conv1d(backend):
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, activation='linear', name='first_layer', input_shape=input_shape_2d(backend)))
    model.add(AveragePooling1D(pool_size=2, name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv1D(4, kernel_size=2, activation='relu', name='middle_layer'))
    model.add(Conv1D(4, kernel_size=1, activation='relu', name='last_layer'))  # Will become PointwiseConv1D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture()
def keras_model_conv2d(backend):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), activation='linear', name='first_layer', input_shape=input_shape_3d(backend)))
    model.add(AveragePooling2D(pool_size=(2, 2), name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', name='middle_layer'))
    model.add(Conv2D(4, kernel_size=(1, 1), activation='relu', name='last_layer'))  # Will become PointwiseConv2D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture()
def keras_model_sepconv1d(backend):
    model = Sequential()
    model.add(
        SeparableConv1D(8, kernel_size=3, activation='linear', name='first_layer', input_shape=input_shape_2d(backend))
    )
    model.add(AveragePooling1D(pool_size=2, name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv1D(4, kernel_size=2, activation='relu', name='middle_layer'))
    model.add(Conv1D(4, kernel_size=1, activation='relu', name='last_layer'))  # Will become PointwiseConv1D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture()
def keras_model_sepconv2d(backend):
    model = Sequential()
    model.add(
        SeparableConv2D(8, kernel_size=(3, 3), activation='linear', name='first_layer', input_shape=input_shape_3d(backend))
    )
    model.add(AveragePooling2D(pool_size=(2, 2), name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', name='middle_layer'))
    model.add(Conv2D(4, kernel_size=(1, 1), activation='relu', name='last_layer'))  # Will become PointwiseConv2D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'XLS'])
@pytest.mark.parametrize('model_type', ['conv1d', 'conv2d'])
def test_auto_precision_conv(
    test_case_id, keras_model_conv1d, keras_model_conv2d, data_2d, data_3d, model_type, io_type, backend
):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')

    if model_type == 'conv1d':
        model = keras_model_conv1d
        data = data_2d
    else:
        model = keras_model_conv2d
        data = data_3d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<16,6>', granularity='model')
    config['LayerName'] = {
        # Infer all types of these layers
        'first_layer': {
            'Precision': 'auto',
        },
        'first_pool': {
            'Precision': 'auto',
        },
        # Infer only a few specific types for these layers
        'middle_layer': {
            'Precision': {
                'accum': 'auto',
                'weight': 'auto',
            },
        },
        'last_layer': {
            'Precision': {
                'result': 'auto',
            },
        },
    }

    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )

    # Compile will fail if there are still UnspecifiedPrecisionTypes in the model
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=2e-2, atol=5e-2, verbose=True)


@pytest.mark.parametrize('io_type', ['io_stream'])  # Until we implement SeparableConv1D/2D for io_parallel
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])  # No SeparableConv1D/2D in Quartus
@pytest.mark.parametrize('model_type', ['sepconv1d', 'sepconv2d'])
def test_auto_precision_sepconv(
    test_case_id, keras_model_sepconv1d, keras_model_sepconv2d, data_2d, data_3d, model_type, io_type, backend
):
    if model_type == 'sepconv1d':
        model = keras_model_sepconv1d
        data = data_2d
    else:
        model = keras_model_sepconv2d
        data = data_3d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<16,6>', granularity='model')
    config['LayerName'] = {
        # Infer all types of these layers
        'first_layer': {
            'Precision': 'auto',
        },
        'first_pool': {
            'Precision': 'auto',
        },
        # Infer only a few specific types for these layers
        'middle_layer': {
            'Precision': {
                'accum': 'auto',
                'weight': 'auto',
            },
        },
        'last_layer': {
            'Precision': {
                'result': 'auto',
            },
        },
    }
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )

    # Compile will fail if there are still UnspecifiedPrecisionTypes in the model
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=2e-2, atol=5e-2, verbose=True)


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'XLS'])
def test_auto_precision_dense(test_case_id, keras_model_dense, data_1d, io_type, backend):
    if backend == 'XLS' and io_type != 'io_parallel':
        pytest.skip(f'XLS backend only supports IOType: io_parallel, but got: {io_type}')

    model = keras_model_dense
    data = data_1d

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<16,6>', granularity='model')
    config['LayerName'] = {
        # Infer all types of these layers
        'first_layer': {
            'Precision': 'auto',
        },
        'first_bn': {
            'Precision': 'auto',
        },
        # Infer only a few specific types for these layers
        'middle_layer': {
            'Precision': {
                'accum': 'auto',
                'weight': 'auto',
            },
        },
        'last_layer': {
            'Precision': {
                'result': 'auto',
            },
        },
    }
    odir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )

    # Compile will fail if there are still UnspecifiedPrecisionTypes in the model
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=2e-2, atol=5e-2, verbose=True)


@pytest.mark.parametrize(
    'val, expected_width',
    [
        (0, 1),
        (-1024, 1),
        (1024, 1),
        (0.03125, 1),
        (-0.03125, 1),
        (1.25, 3),
        (-1.25, 4),
        (1.1, 8),
        (-1.1, 9),
    ],
)
def test_precision_from_constant_unit(val, expected_width):
    """Test determining precision needed for a constant."""
    max_width = 8
    fp = _get_precision_from_constant(val, max_width)

    assert fp.min <= val <= fp.max
    assert fp.width == expected_width
    assert fp.signed == (val < 0)

    quantum = 2.0**-fp.fractional
    if expected_width < max_width:
        assert val % quantum == 0
