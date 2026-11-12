import numpy as np
import pytest

from hls4ml.model.layers import Activation, ParametrizedActivation
from hls4ml.model.optimizer.passes import bit_exact


@pytest.fixture
def input_kif(monkeypatch):
    value = (
        np.array([1, 0], dtype=np.int16),
        np.array([2, 2], dtype=np.int16),
        np.array([3, 3], dtype=np.int16),
    )
    monkeypatch.setattr(bit_exact, 'get_input_kifs', lambda _: (tuple(v.copy() for v in value),))
    return value


def make_layer(layer_type, activation, **attributes):
    layer = object.__new__(layer_type)
    layer.attributes = {'activation': activation, **attributes}
    return layer


def test_produce_kif_accepts_keras_leaky_relu_name(input_kif):
    """Keras uses ``leaky_relu`` rather than the legacy ``leakyrelu`` name."""
    layer = make_layer(ParametrizedActivation, 'leaky_relu', activ_param=0.25)

    k, i, f = bit_exact._produce_kif(layer)

    np.testing.assert_array_equal(k, input_kif[0])
    np.testing.assert_array_equal(i, input_kif[1])
    np.testing.assert_array_equal(f, input_kif[2] + 2)


@pytest.mark.parametrize(
    'layer',
    [make_layer(Activation, 'unknown'), make_layer(ParametrizedActivation, 'unknown', activ_param=0.25)],
    ids=['activation', 'parametrized_activation'],
)
def test_produce_kif_fallback_preserves_shape_and_dtype(input_kif, layer):
    k, i, f = bit_exact._produce_kif(layer)

    for value in (k, i, f):
        assert value.shape == input_kif[0].shape
        assert value.dtype == np.int16
