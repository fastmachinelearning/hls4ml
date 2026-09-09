from pathlib import Path

import numpy as np
import pytest
from quantizers.fixed_point.fixed_point_ops_np import get_fixed_quantizer_np, saturation_mode_registry_np
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.model.types import FixedPrecisionType

test_root_path = Path(__file__).parent


# Test all rounding and saturation modes in XLS using a linear activation layer with low output precision
@pytest.mark.parametrize('backend', ['XLS'])
@pytest.mark.parametrize('round_mode', ['TRN', 'TRN_ZERO', 'RND', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF', 'RND_CONV'])
@pytest.mark.parametrize('sat_mode', ['WRAP', 'SAT', 'SAT_ZERO', 'SAT_SYM'])
def test_rounding_saturation(test_case_id, round_mode, sat_mode, backend):
    if sat_mode == 'SAT_ZERO' and 'SAT_ZERO' not in saturation_mode_registry_np:
        # TODO remove this skip and update quantizers version once quantizers support SAT_ZERO,
        # see https://github.com/calad0i/quantizers/pull/1
        pytest.skip('SAT_ZERO is not supported by the installed quantizers package')

    in_precision = FixedPrecisionType(width=8, integer=4, signed=True)
    out_precision = FixedPrecisionType(width=4, integer=3, signed=True, rounding_mode=round_mode, saturation_mode=sat_mode)
    # out_precision covers the range [-4, -3.5, ... 3.5]
    # Input data a range near the boundaries and near zero with finer resolution = 0.125,
    # allowing to cover all saturation and rounding cases.
    step = 2 ** (-out_precision.fractional - 2)
    X = np.append(
        np.arange(-step * 4, step * 4, step),
        [
            np.arange(out_precision.min - 1, out_precision.min + 1, step),
            np.arange(out_precision.max - 1, out_precision.max + 1, step),
        ],
    )

    input = Input(shape=X.shape)
    activation = Activation('linear', name='activation')(input)
    keras_model = Model(inputs=input, outputs=activation)

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', backend=backend)
    hls_config['Model']['Precision'] = in_precision
    hls_config['LayerName']['activation']['Precision']['result'] = out_precision

    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_config, io_type='io_parallel', output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X)

    quantizer = get_fixed_quantizer_np(round_mode, sat_mode)
    expected_prediction = quantizer(X, k=1, i=out_precision.integer - 1, f=out_precision.fractional)

    np.testing.assert_equal(hls_prediction, expected_prediction)
