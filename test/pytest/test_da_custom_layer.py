from pathlib import Path

import keras
import numpy as np
import pytest
from hgq.config import QuantizerConfigScope
from hgq.layers import QGRU, QDenseT

from hls4ml.converters import convert_from_keras_model

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('return_sequences', [True, False])
def test_qgru(return_sequences, test_case_id):
    with QuantizerConfigScope(b0=4, i0=2):
        inp = keras.Input(shape=(8, 4))
        out = QGRU(4, return_sequences=return_sequences, return_state=False)(inp)
        model = keras.Model(inputs=inp, outputs=out)

    model_hls = convert_from_keras_model(model, output_dir=str(test_root_path / test_case_id), backend='Vitis')
    model_hls.compile()

    data_in = np.random.rand(1000, 8, 4).astype(np.float32) * 32 - 16
    data_out = model.predict(data_in, batch_size=1000)
    data_out_hls = model_hls.predict(data_in).reshape(data_out.shape)  # type: ignore

    np.testing.assert_equal(data_out, data_out_hls)


def test_qdense_t(test_case_id):
    with QuantizerConfigScope(b0=4, i0=2):
        inp = keras.Input(shape=(8,))
        out = QDenseT(4)(inp)
        model = keras.Model(inputs=inp, outputs=out)

    model_hls = convert_from_keras_model(model, output_dir=str(test_root_path / test_case_id), backend='Vitis')
    model_hls.compile()

    data_in = np.random.rand(1000, 8).astype(np.float32) * 32 - 16
    data_out = model.predict(data_in, batch_size=1000)
    data_out_hls = model_hls.predict(data_in)

    np.testing.assert_equal(data_out, data_out_hls)
