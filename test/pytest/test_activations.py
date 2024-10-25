from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import ELU, Activation, Input, LeakyReLU, ReLU, ThresholdedReLU
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent

# Variable 'name' is simply used as an identifier for the activation


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('shape, io_type', [((8,), 'io_parallel'), ((8,), 'io_stream'), ((8, 8, 3), 'io_stream')])
@pytest.mark.parametrize(
    'activation, name',
    [
        (ReLU(), 'relu'),
        (LeakyReLU(alpha=1.5), 'leaky_relu'),
        (Activation('leaky_relu'), 'leaky_relu_act'),
        (ThresholdedReLU(theta=0.75), 'threshold_relu'),
        (ELU(alpha=1.25), 'elu'),
        (Activation('selu'), 'selu'),
        # Tensorflow exception of multi-dimensional PReLU (8, 8, 3)
        # (PReLU(alpha_initializer='zeros'), 'prelu'),
        (Activation('softplus'), 'softplus'),
        (Activation('softsign'), 'softsign'),
        (Activation(activation='tanh'), 'tanh'),
        (Activation('sigmoid'), 'sigmoid'),
        # Theano and Tensorflow might have different definitions for hard sigmoid
        # Result is likely to be different when |x| > 1 (see TF/Theano docs)
        (Activation('hard_sigmoid'), 'hard_sigmoid'),
    ],
)
def test_activations(backend, activation, name, shape, io_type):
    # Subtract 0.5 to include negative values
    X = np.random.rand(1000, *shape) - 0.5

    input = Input(shape=shape)
    activation = activation(input)
    keras_model = Model(inputs=input, outputs=activation)

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', backend=backend)
    output_dir = str(test_root_path / 'hls4mlprj_activations_{}_{}_{}_{}').format(backend, io_type, str(shape), name)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_config, io_type=io_type, output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    keras_prediction = keras_model.predict(X)
    hls_prediction = hls_model.predict(X).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=2e-2, atol=2e-2)
