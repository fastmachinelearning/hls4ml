from pathlib import Path

import numpy as np
import pytest
from qkeras import QActivation, QBatchNormalization, QConv2D, QDense
from tensorflow.keras.layers import Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize(
    'backend,io_type,strategy',
    [
        ('Quartus', 'io_parallel', 'resource'),
        ('Quartus', 'io_stream', 'resource'),
        ('Vivado', 'io_parallel', 'resource'),
        ('Vivado', 'io_parallel', 'latency'),
        ('Vivado', 'io_stream', 'latency'),
        ('Vivado', 'io_stream', 'resource'),
        ('Vitis', 'io_parallel', 'resource'),
        ('Vitis', 'io_parallel', 'latency'),
        ('Vitis', 'io_stream', 'latency'),
        ('Vitis', 'io_stream', 'resource'),
    ],
)
def test_binary_cnn(backend, io_type, strategy):
    x_in = Input(shape=(28, 28, 1))

    x = QConv2D(
        4,
        (3, 3),
        kernel_quantizer='binary',
        name='conv2d_1',
        kernel_regularizer=l2(0.0001),
        use_bias=True,
        bias_quantizer='quantized_bits(5,2)',
    )(x_in)
    x = QBatchNormalization()(x)
    x = QActivation('binary', name='act1')(x)

    x = QConv2D(8, (3, 3), kernel_quantizer='binary', name='conv2d_2', kernel_regularizer=l2(0.0001), use_bias=False)(x)
    x = QBatchNormalization()(x)
    x = QActivation('binary', name='act2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = QConv2D(8, (3, 3), kernel_quantizer='binary', name='conv2d_3', kernel_regularizer=l2(0.0001), use_bias=False)(x)
    x = QBatchNormalization()(x)
    x = QActivation('binary', name='act3')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = QDense(10, kernel_quantizer='binary', name='q_dense_6', use_bias=False)(x)
    x = QBatchNormalization()(x)
    x = QActivation('binary_tanh', name='act4')(x)

    x = QDense(10, kernel_quantizer='binary', activation='linear', name='q_dense_7', use_bias=False)(x)

    model2 = Model(inputs=x_in, outputs=x)

    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model2.summary()

    hls_config = hls4ml.utils.config_from_keras_model(
        model2, granularity='name', default_precision='fixed<32,12>', backend=backend
    )
    hls_config['Model']['Strategy'] = strategy

    # hls_config['LayerName']['q_dense_7_softmax']['Implementation'] = 'legacy'

    hls_config['LayerName']['conv2d_1']['ReuseFactor'] = 9
    hls_config['LayerName']['conv2d_2']['ReuseFactor'] = 36
    hls_config['LayerName']['conv2d_3']['ReuseFactor'] = 72
    hls_config['LayerName']['q_dense_6']['ReuseFactor'] = 2000
    hls_config['LayerName']['q_dense_7']['ReuseFactor'] = 100

    if backend == 'Quartus' and io_type == 'io_parallel':
        # Winegrad imp[lementation does not support binary
        hls_config['LayerName']['conv2d_1']['Implementation'] = 'im2col'
        hls_config['LayerName']['conv2d_2']['Implementation'] = 'im2col'
        hls_config['LayerName']['conv2d_3']['Implementation'] = 'im2col'

    output_dir = str(test_root_path / f'hls4mlprj_binary_cnn_{backend}_{io_type}_{strategy}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model2,
        hls_config=hls_config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    X = np.random.rand(100, 28, 28, 1)
    X = np.round(X * 2**10) * 2**-10

    hls_model.compile()
    y = model2.predict(X)  # noqa: F841
    y_hls = hls_model.predict(X)  # noqa: F841

    np.testing.assert_allclose(y_hls, y, rtol=1e-2, atol=0.01)
