from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

# backends = ['Vivado', 'Vitis']
backends = ['Vitis']

import os

os.environ['XILINX_VITIS'] = "/opt/Xilinx/Vitis_HLS/2023.2/"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']


# @pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backends)
def test_fifo_depth(backend):

    input_shape = (128, 128, 3)
    activation = 'relu'
    kernel_size = (3, 3)
    padding = 'same'

    model = Sequential()
    model.add(
        SeparableConv2D(filters=4, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape)
    )
    model.add(SeparableConv2D(filters=8, kernel_size=kernel_size, padding=padding, activation=activation))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<8, 4>')
    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=200_000)

    output_dir = str(test_root_path / f'hls4mlprj_fifo_depth_optimization_backend_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type='io_stream', hls_config=config, output_dir=output_dir, backend=backend
    )

    hls_model.build(reset=False, csim=False, synth=True, cosim=True)
    hls4ml.report.read_vivado_report(output_dir)

    # config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>')
    # output_dir = str(
    #     test_root_path
    #     / 'hls4mlprj_fifo_depth_optimization_backend_{}'.format(
    #         backend
    #     )
    # )
    # hls_model = hls4ml.converters.convert_from_keras_model(
    #     model, hls_config=config, output_dir=output_dir, io_type='io_stream', backend=backend
    # )
    # hls_model.compile()
    # hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    # np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)
