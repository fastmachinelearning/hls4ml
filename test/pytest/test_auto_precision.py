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

test_root_path = Path(__file__).parent

in_height = 10
in_width = 12
in_feat = 4


@pytest.fixture(scope='module')
def data_1d():
    X = np.random.rand(100, in_feat)
    return X


@pytest.fixture(scope='module')
def data_2d():
    X = np.random.rand(100, in_width, in_feat)
    return X


@pytest.fixture(scope='module')
def data_3d():
    X = np.random.rand(100, in_height, in_width, in_feat)
    return X


@pytest.fixture(scope='module')
def keras_model_dense():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(in_feat,), name='first_layer'))
    model.add(BatchNormalization(name='first_bn'))
    model.add(Dense(6, activation='relu', name='middle_layer'))
    model.add(BatchNormalization(name='middle_bn'))
    model.add(Dense(4, activation='relu', name='last_layer'))
    model.compile()
    return model


@pytest.fixture(scope='module')
def keras_model_conv1d():
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, activation='linear', name='first_layer', input_shape=(in_width, in_feat)))
    model.add(AveragePooling1D(pool_size=2, name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv1D(4, kernel_size=2, activation='relu', name='middle_layer'))
    model.add(Conv1D(4, kernel_size=1, activation='relu', name='last_layer'))  # Will become PointwiseConv1D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture(scope='module')
def keras_model_conv2d():
    model = Sequential()
    model.add(
        Conv2D(8, kernel_size=(3, 3), activation='linear', name='first_layer', input_shape=(in_height, in_width, in_feat))
    )
    model.add(AveragePooling2D(pool_size=(2, 2), name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', name='middle_layer'))
    model.add(Conv2D(4, kernel_size=(1, 1), activation='relu', name='last_layer'))  # Will become PointwiseConv2D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture(scope='module')
def keras_model_sepconv1d():
    model = Sequential()
    model.add(SeparableConv1D(8, kernel_size=3, activation='linear', name='first_layer', input_shape=(in_width, in_feat)))
    model.add(AveragePooling1D(pool_size=2, name='first_pool'))
    model.add(ReLU(name='first_act'))
    model.add(Conv1D(4, kernel_size=2, activation='relu', name='middle_layer'))
    model.add(Conv1D(4, kernel_size=1, activation='relu', name='last_layer'))  # Will become PointwiseConv1D
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.compile()
    return model


@pytest.fixture(scope='module')
def keras_model_sepconv2d():
    model = Sequential()
    model.add(
        SeparableConv2D(
            8, kernel_size=(3, 3), activation='linear', name='first_layer', input_shape=(in_height, in_width, in_feat)
        )
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
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('model_type', ['conv1d', 'conv2d'])
def test_auto_precision_conv(keras_model_conv1d, keras_model_conv2d, data_2d, data_3d, model_type, io_type, backend):
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

    odir = str(test_root_path / f'hls4mlprj_auto_{model_type}_{backend}_{io_type}')
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
    keras_model_sepconv1d, keras_model_sepconv2d, data_2d, data_3d, model_type, io_type, backend
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
    odir = str(test_root_path / f'hls4mlprj_auto_{model_type}_{backend}_{io_type}')
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
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_auto_precision_dense(keras_model_dense, data_1d, io_type, backend):
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
    odir = str(test_root_path / f'hls4mlprj_auto_dense_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )

    # Compile will fail if there are still UnspecifiedPrecisionTypes in the model
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=2e-2, atol=5e-2, verbose=True)
