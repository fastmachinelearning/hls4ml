from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import hls4ml

test_root_path = Path(__file__).parent


class InstanceNormalization(keras.layers.Layer):
    """Minimal reimplementation of ``tensorflow_addons.layers.InstanceNormalization``.

    TensorFlow-Addons is archived and does not support Keras 3, so the layer is reproduced here
    (following the original implementation) to test the hls4ml support for the layer class.
    """

    def __init__(self, axis, epsilon=1e-3, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = list(axis)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = int(input_shape[-1])
        if self.scale:
            self.gamma = self.add_weight(name='gamma', shape=(dim,), initializer='ones')
        if self.center:
            self.beta = self.add_weight(name='beta', shape=(dim,), initializer='zeros')

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
        result = tf.math.rsqrt(variance + self.epsilon) * (inputs - mean)
        if self.scale:
            result = result * self.gamma
        if self.center:
            result = result + self.beta
        return result

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale})
        return config


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_instancenorm(test_case_id, backend):
    """1D spatial input, i.e. shape (seq_len, n_chan)"""
    in_shape = (16, 8)
    model = keras.Sequential()
    model.add(keras.Input(shape=in_shape))
    model.add(InstanceNormalization(axis=[1]))
    model.compile()

    np.random.seed(0)
    data = np.random.rand(100, *in_shape)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    y_keras = model.predict(data, verbose=0).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=2e-2, verbose=True)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_instancenorm_2d(test_case_id, backend):
    """2D spatial input, i.e. shape (height, width, n_chan)"""
    in_shape = (6, 8, 4)
    model = keras.Sequential()
    model.add(keras.Input(shape=in_shape))
    model.add(InstanceNormalization(axis=[1, 2], epsilon=1e-2))
    gamma = np.random.normal(1.0, 0.1, size=(in_shape[-1],))
    beta = np.random.normal(0.0, 0.1, size=(in_shape[-1],))
    model.layers[0].set_weights([gamma, beta])
    model.compile()

    np.random.seed(0)
    data = np.random.rand(100, *in_shape)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    y_keras = model.predict(data, verbose=0).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=2e-2, verbose=True)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_instancenorm_no_center_scale(test_case_id, backend):
    in_shape = (10, 6)
    model = keras.Sequential()
    model.add(keras.Input(shape=in_shape))
    model.add(InstanceNormalization(axis=[1], center=False, scale=False))
    model.compile()

    np.random.seed(0)
    data = np.random.rand(100, *in_shape)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    y_keras = model.predict(data, verbose=0).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=2e-2, verbose=True)


def test_instancenorm_rejects_channel_axis(test_case_id):
    """Statistics over the channel axis (per-position normalization) are not supported."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(8, 4)))
    model.add(InstanceNormalization(axis=[2]))
    model.compile()

    with pytest.raises(NotImplementedError, match='spatial'):
        config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
        output_dir = str(test_root_path / test_case_id)
        hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='Vivado', output_dir=output_dir)
