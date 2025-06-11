from pathlib import Path

import numpy as np
import pytest
from keras.layers import Add, Dense
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def model_clone_precision_inheritance():
    inp = keras.Input(shape=(10,), name='inp')
    x = Dense(10, name='x')(inp)
    y = Dense(10, name='y')(inp)
    out = Add(name='out')([x, y])
    model = keras.Model(inp, out)
    return model


@pytest.fixture(scope='module')
def model_multi_clone():
    # inp will be cloned 7 times
    # a will be cloned 6 times
    # ab will be cloned 5 times
    # abac will be cloned 4 times
    # abacabad will be cloned 3 times
    # abacabadabacabae will be cloned 2 times

    inp = keras.Input(shape=(10,))
    a = Dense(10, name='dense_5_copies')(inp)
    b = Dense(10)(inp)
    c = Dense(10)(inp)
    d = Dense(10)(inp)
    e = Dense(10)(inp)
    f = Dense(10)(inp)
    g = Dense(10)(inp)
    ab = Add(name='add_4_copies')([a, b])
    ac = Add()([a, c])
    ad = Add()([a, d])
    ae = Add()([a, e])
    af = Add()([a, f])
    ag = Add()([a, g])
    abac = Add(name='add_3_copies')([ab, ac])
    abad = Add()([ab, ad])
    abae = Add()([ab, ae])
    abaf = Add()([ab, af])
    abag = Add()([ab, ag])
    abacabad = Add(name='add_2_copies')([abac, abad])
    abacabae = Add()([abac, abae])
    abacabaf = Add()([abac, abaf])
    abacabag = Add()([abac, abag])
    abacabadabacabae = Add()([abacabad, abacabae])
    abacabadabacabaf = Add()([abacabad, abacabaf])
    abacabadabacabag = Add()([abacabad, abacabag])
    abacabadabacabaeabacabadabacabaf = Add()([abacabadabacabae, abacabadabacabaf])
    abacabadabacabaeabacabadabacabag = Add()([abacabadabacabae, abacabadabacabag])
    out = Add()([abacabadabacabaeabacabadabacabaf, abacabadabacabaeabacabadabacabag])
    model = keras.Model(inp, out)
    return model


@pytest.fixture(scope='module')
def data():
    X = np.random.normal(0, 0.2, (100, 10))
    X = np.clip(X, -2, 2)
    return X


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis', 'oneAPI'])
def test_multi_clone(model_multi_clone, data, backend: str):
    output_dir = str(test_root_path / f'hls4mlprj_stream_clone_multiclone_{backend}')
    hls_config = {'Model': {'Precision': 'fixed<32,5>', 'ReuseFactor': 1}}
    model_hls = convert_from_keras_model(
        model_multi_clone,
        backend=backend,
        output_dir=output_dir,
        hls_config=hls_config,
        io_type='io_stream',  # clone only happens with stream io.
    )
    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = model_multi_clone(data).numpy()

    assert np.allclose(r_hls, r_keras, atol=1e-5, rtol=0)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus', 'Vitis', 'oneAPI'])
def test_clone_precision_inheritance(model_clone_precision_inheritance, data, backend: str):
    output_dir = str(test_root_path / f'hls4mlprj_stream_clone_precision_{backend}')
    layer_config = {
        'inp': {'Precision': 'fixed<32,5>'},
        'x': {'Precision': 'fixed<32,5>'},
        'x_linear': {'Precision': 'fixed<32,5>'},
        'y': {'Precision': 'fixed<32,5>'},
        'y_linear': {'Precision': 'fixed<32,5>'},
        'out': {'Precision': 'fixed<32,5>'},
    }
    hls_config = {'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}, 'LayerName': layer_config}
    model_hls = convert_from_keras_model(
        model_clone_precision_inheritance,
        backend=backend,
        output_dir=output_dir,
        hls_config=hls_config,
        io_type='io_stream',  # clone only happens with stream io.
    )
    assert model_hls.graph['clone_inp'].attributes['inp_cpy1'].type.precision.width == 32
    assert model_hls.graph['clone_inp'].attributes['inp_cpy1'].type.precision.integer == 5
    assert model_hls.graph['clone_inp'].attributes['inp_cpy2'].type.precision.width == 32
    assert model_hls.graph['clone_inp'].attributes['inp_cpy2'].type.precision.integer == 5

    model_hls.compile()
    r_hls = model_hls.predict(data)
    r_keras = model_clone_precision_inheritance(data).numpy()

    assert np.allclose(r_hls, r_keras, atol=1e-5, rtol=0)


if __name__ == '__main__':
    test_clone_precision_inheritance(model_clone_precision_inheritance(), data(), 'Vivado')
