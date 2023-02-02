from pathlib import Path

import pytest
from tensorflow.keras.layers import Concatenate, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope="module")
def keras_model():
    inp1 = Input(shape=(2, 3), name="input_1")
    x = MaxPooling1D()(inp1)
    x = Flatten()(x)
    y = Flatten()(inp1)
    out = Concatenate(axis=1)([x, y])
    model = Model(inputs=inp1, outputs=out)
    return model


@pytest.fixture
@pytest.mark.parametrize("io_type", ["io_stream"])
@pytest.mark.parametrize("backend", ["Vivado"])
def hls_model(keras_model, backend, io_type):
    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model,
        default_precision="ap_int<16>",
        granularity="name",
    )
    output_dir = str(test_root_path / f"hls4mlprj_clone_flatten_{backend}_{io_type}")
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=hls_config,
        io_type=io_type,
        backend=backend,
        output_dir=output_dir,
    )

    hls_model.compile()
    return hls_model
