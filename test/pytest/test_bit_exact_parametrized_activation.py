import numpy as np

from hls4ml.model.layers import ParametrizedActivation
from hls4ml.model.optimizer.passes import bit_exact


def test_produce_kif_accepts_keras_leaky_relu_name(monkeypatch):
    """Keras uses ``leaky_relu`` rather than the legacy ``leakyrelu`` name."""
    input_kif = (
        np.array([1, 0], dtype=np.int16),
        np.array([2, 2], dtype=np.int16),
        np.array([3, 3], dtype=np.int16),
    )
    monkeypatch.setattr(bit_exact, 'get_input_kifs', lambda _: (input_kif,))

    def produce(activation_name):
        layer = object.__new__(ParametrizedActivation)
        layer.attributes = {'activation': activation_name, 'activ_param': 0.25}
        return bit_exact._produce_kif(layer)

    legacy_result = produce('leakyrelu')
    keras_result = produce('leaky_relu')

    for legacy_value, keras_value in zip(legacy_result, keras_result):
        np.testing.assert_array_equal(keras_value, legacy_value)
