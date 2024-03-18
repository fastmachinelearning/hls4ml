""" Test that nested models in Keras is properly parsed and expanded by the optimizers.
"""

from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential

import hls4ml

test_root_path = Path(__file__).parent


def make_nested_model(input_shape):
    """
    This model will have the following architecture:
    Functional (fun_model)
        Dense (fun_first_dense)
        Sequential (seq_sub)
            Dense
            Dense
        Dense (fun_middle_dense)
        Functional (fun_sub)
            Dense
            Dense
        Dense (fun_last_dense)
    """
    seq_sub = Sequential(name='seq_sub')
    seq_sub.add(Dense(5, activation='linear', input_shape=(5,), name='seq_sub_dense_1'))
    seq_sub.add(Dense(3, activation='linear', name='seq_sub_dense_2'))

    fun_input = Input(shape=(8,), name='fun_input')
    fun_x = Dense(7, activation='linear', name='fun_sub_dense_1')(fun_input)
    fun_x = Dense(6, activation='linear', name='fun_sub_dense_2')(fun_x)
    fun_sub = Model(inputs=fun_input, outputs=fun_x, name='fun_sub')

    input = Input(shape=input_shape, name='model_input')
    x = Dense(5, activation='linear', name='fun_first_dense')(input)
    x = seq_sub(x)
    x = Dense(8, activation='linear', name='fun_middle_dense')(x)
    x = fun_sub(x)
    x = Dense(4, activation='linear', name='fun_last_dense')(x)
    fun_model = Model(inputs=input, outputs=x, name='fun_model')

    return fun_model


def make_sub_nested_model(input_shape):
    """
    The following abomination will create this hierarchy:
    Sequential
        Dense (first_dense)
        Functional (fun_model)
            Dense (fun_first_dense)
            Sequential (fun_model_seq_sub)
                Dense
                Dense
            Dense (fun_middle_dense)
            Functional (fun_model_fun_sub)
                Dense
                Dense
            Dense (fun_last_dense)
        Dense (middle_dense)
        Sequential (seq_model)
            Dense
            Functional (seq_model_fun_sub)
                Dense
                Dense
            Dense
            Sequential (seq_model_seq_sub)
                Dense
                Dense
            Dense
        Dense (last_dense)
    """
    fun_model_seq_sub = Sequential(name='fun_model_seq_sub')
    fun_model_seq_sub.add(Dense(5, activation='linear', input_shape=(5,), name='fun_seq_sub_dense_1'))
    fun_model_seq_sub.add(Dense(3, activation='linear', name='fun_seq_sub_dense_2'))

    fun_fun_input = Input(shape=(8,), name='fun_fun_input')
    fun_fun_x = Dense(7, activation='linear', name='fun_fun_sub_dense_1')(fun_fun_input)
    fun_fun_x = Dense(6, activation='linear', name='fun_fun_sub_dense_2')(fun_fun_x)
    fun_model_fun_sub = Model(inputs=fun_fun_input, outputs=fun_fun_x, name='fun_model_fun_sub')

    fun_input = Input(shape=(10,), name='fun_input')
    fun_x = Dense(5, activation='linear', name='fun_first_dense')(fun_input)
    fun_x = fun_model_seq_sub(fun_x)
    fun_x = Dense(8, activation='linear', name='fun_middle_dense')(fun_x)
    fun_x = fun_model_fun_sub(fun_x)
    fun_x = Dense(4, activation='linear', name='fun_last_dense')(fun_x)
    fun_model = Model(inputs=fun_input, outputs=fun_x, name='fun_model')

    seq_fun_input = Input(shape=(2,), name='seq_fun_input')
    seq_fun_x = Dense(9, activation='linear', name='seq_fun_sub_dense_1')(seq_fun_input)
    seq_fun_x = Dense(3, activation='linear', name='seq_fun_sub_dense_2')(seq_fun_x)
    seq_model_fun_sub = Model(inputs=seq_fun_input, outputs=seq_fun_x, name='seq_model_fun_sub')

    seq_model_seq_sub = Sequential(name='seq_model_seq_sub')
    seq_model_seq_sub.add(Dense(5, activation='linear', input_shape=(2,), name='seq_seq_sub_dense_1'))
    seq_model_seq_sub.add(Dense(7, activation='linear', name='seq_seq_sub_dense_2'))

    seq_model = Sequential(name='seq_model')
    seq_model.add(Dense(2, activation='linear', input_shape=(6,), name='seq_first_dense'))
    seq_model.add(seq_model_fun_sub)
    seq_model.add(Dense(2, activation='linear', name='seq_middle_dense'))
    seq_model.add(seq_model_seq_sub)
    seq_model.add(Dense(2, activation='linear', name='seq_last_dense'))

    model = Sequential()
    model.add(Dense(10, activation='linear', input_shape=input_shape, name='first_dense'))
    model.add(fun_model)
    model.add(Dense(6, activation='linear', name='middle_dense'))
    model.add(seq_model)
    model.add(Dense(4, activation='linear', name='last_dense'))

    return model


def randX(batch_size, N):
    return np.random.rand(batch_size, N)


@pytest.fixture(scope='module')
def randX_20_15():
    return randX(20, 15)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_nested_model(randX_20_15, backend, io_type):
    n_in = 15
    input_shape = (n_in,)
    keras_model = make_nested_model(input_shape)
    keras_model.compile(optimizer='adam', loss='mae')

    config = hls4ml.utils.config_from_keras_model(keras_model, default_precision='fixed<24,12>')
    prj_name = f'hls4mlprj_nested_model_{backend}_{io_type}'
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X = randX_20_15
    y_keras = keras_model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_keras.ravel(), y_hls4ml.ravel(), rtol=1e-2, atol=0.02)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_sub_nested_model(randX_20_15, backend, io_type):
    n_in = 15
    input_shape = (n_in,)
    keras_model = make_sub_nested_model(input_shape)
    keras_model.compile(optimizer='adam', loss='mae')

    config = hls4ml.utils.config_from_keras_model(keras_model, default_precision='fixed<24,12>')
    prj_name = f'hls4mlprj_sub_nested_model_{backend}_{io_type}'
    output_dir = str(test_root_path / prj_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    hls_model.compile()

    X = randX_20_15
    y_keras = keras_model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_keras.ravel(), y_hls4ml.ravel(), rtol=1e-2, atol=0.02)
