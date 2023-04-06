import numpy as np
import pytest
from qkeras import QActivation, QBatchNormalization, QConv2D, QDense
from tensorflow.keras.layers import Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import hls4ml


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_model2(backend, io_type):
    x_in = Input(shape=(28, 28, 1))

    x = QConv2D(
        4,
        (3, 3),
        kernel_quantizer="binary",
        bias_quantizer="binary",
        name="conv2d_1",
        kernel_regularizer=l2(0.0001),
    )(x_in)
    x = QBatchNormalization()(x)
    x = QActivation("binary", name="act1")(x)

    x = QConv2D(
        8,
        (3, 3),
        kernel_quantizer="binary",
        bias_quantizer="binary",
        name="conv2d_2",
        kernel_regularizer=l2(0.0001),
    )(x)
    x = QBatchNormalization()(x)
    x = QActivation("binary", name="act2")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = QConv2D(
        8,
        (3, 3),
        kernel_quantizer="binary",
        bias_quantizer="binary",
        name="conv2d_3",
        kernel_regularizer=l2(0.0001),
    )(x)
    x = QBatchNormalization()(x)
    x = QActivation("binary", name="act3")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = QDense(10, kernel_quantizer="binary", bias_quantizer="binary", name="q_dense_6")(x)
    x = QBatchNormalization()(x)
    x = QActivation("binary_tanh", name="act4")(x)

    x = QDense(
        10,
        kernel_quantizer="binary",
        bias_quantizer="binary",
        activation="softmax",
        name="q_dense_7",
    )(x)

    model2 = Model(inputs=x_in, outputs=x)

    model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model2.summary()

    hls_config = hls4ml.utils.config_from_keras_model(model2, granularity="name")
    hls_config["Model"]["Strategy"] = "Resource"
    hls_config["IOTtype"] = "io_stream"

    hls_config["LayerName"]["conv2d_1"]["ReuseFactor"] = 36
    hls_config["LayerName"]["conv2d_2"]["ReuseFactor"] = 288
    hls_config["LayerName"]["conv2d_3"]["ReuseFactor"] = 576
    hls_config["LayerName"]["q_dense_6"]["ReuseFactor"] = 2000
    hls_config["LayerName"]["q_dense_7"]["ReuseFactor"] = 100

    hls_model = hls4ml.converters.convert_from_keras_model(
        model2,
        hls_config=hls_config,
        output_dir=f"hls4mlprj_binary_cnn_{backend}_{io_type}",
        backend=backend,
        io_type=io_type,
    )

    hls_model.compile()
    y = model2.predict(np.zeros((1, 28, 28, 1)))
    y_hls = hls_model.predict(np.zeros((1, 28, 28, 1)))

    print(f"{y_hls=}")
    print(f"{y=}")

    np.testing.assert_allclose(np.squeeze(y_hls), np.squeeze(y), rtol=1e-2, atol=0.01)


if __name__ == "__main__":
    test_model2("Vivado", "io_stream")
