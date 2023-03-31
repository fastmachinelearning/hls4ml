from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_test, y_test = x_test[:1000], y_test[:1000]
    return x_train, y_train, x_test, y_test


@pytest.fixture(scope='module')
def keras_model(mnist_data):
    # Aim of this model is to test different CNN paramaters, including:
    # The common filter sizes, 3x3 and 5x5
    # A non-power of 2 number of filters
    # Both Average and Max Pooling
    # Both Same and Valid Padding
    x_train, y_train, x_test, y_test = mnist_data
    keras_model = Sequential()
    keras_model.add(Conv2D(4, (3, 3), input_shape=(28, 28, 1), padding='same'))
    keras_model.add(Activation('relu'))
    keras_model.add(MaxPooling2D())
    keras_model.add(Conv2D(6, (5, 5), padding='valid'))
    keras_model.add(Activation('relu'))
    keras_model.add(AveragePooling2D())
    keras_model.add(Flatten())
    keras_model.add(Dense(10, kernel_initializer='lecun_uniform'))
    keras_model.add(Activation('softmax', name='softmax'))
    keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras_model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
    return keras_model


@pytest.mark.parametrize(
    'backend,io_type,strategy',
    [
        ('Quartus', 'io_parallel', 'resource'),
        ('Quartus', 'io_stream', 'resource'),
        ('Vivado', 'io_parallel', 'resource'),
        ('Vivado', 'io_parallel', 'latency'),
        ('Vivado', 'io_stream', 'latency'),
        ('Vivado', 'io_stream', 'resource'),
        ('Vitis', 'io_parallel', 'resource'),
        ('Vitis', 'io_parallel', 'latency'),
        ('Vitis', 'io_stream', 'latency'),
        ('Vitis', 'io_stream', 'resource'),
    ],
)
def test_mnist_cnn(keras_model, mnist_data, backend, io_type, strategy):
    x_train, y_train, x_test, y_test = mnist_data

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', backend=backend)
    hls_config['Model']['Strategy'] = strategy
    hls_config['LayerName']['softmax']['Implementation'] = 'stable'
    output_dir = str(test_root_path / f'hls4mlprj_cnn_mnist_{backend}_{io_type}_{strategy}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    # Model under test predictions and accuracy
    y_keras = keras_model.predict(x_test)
    y_hls4ml = hls_model.predict(x_test)

    acc_keras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
    acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls4ml, axis=1))
    rel_diff = abs(acc_keras - acc_hls4ml) / acc_keras

    print(f'Accuracy keras:      {acc_keras}')
    print(f'Accuracy hls4ml:     {acc_hls4ml}')
    print(f'Relative difference: {rel_diff}')

    assert acc_keras > 0.95 and rel_diff < 0.03
