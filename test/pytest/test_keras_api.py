import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Dense,
)

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_dense(backend, io_type):
    model = tf.keras.models.Sequential()
    model.add(
        Dense(
            2,
            input_shape=(1,),
            name='Dense',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
    )
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, 1)

    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_dense_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    vivado_bin_dir = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/opt/Xilinx/Vivado/2020.1/bin'
    os.environ['PATH'] += os.pathsep + vivado_bin_dir
    os.environ['XILINX_VIVADO'] = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1/opt/Xilinx/Vivado/2020.1'

    base_path = '/cvmfs/projects.cern.ch/hls4ml/vivado/2020.1_v1/vivado-2020.1_v1'
    vitis_path = "/opt/Xilinx/Vitis/2020.1"
    original_paths = (
        "/opt/Xilinx/Vitis/2020.1/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/microblaze/lin/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/arm/lin/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/microblaze/linux_toolchain/lin64_le/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-none/bin:"
        + "/opt/Xilinx/Vitis/2020.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:"
        + "/opt/Xilinx/Vitis/2020.1/tps/lnx64/cmake-3.3.2/bin:"
        + "/opt/Xilinx/Vitis/2020.1/cardano/bin"
    )

    update_environment(base_path, original_paths, vitis_path)

    data = hls_model.build()
    print(data)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0]._name
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
    assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]


def compare_synthesis(data, filename):
    with open(filename, "w") as fp:
        baseline = json.dump(data, fp)
    if data == baseline:
        return True
    else:
        return False


def update_environment(base_path, original_paths, vivado_path):
    # Append the new paths to the PATH environment variable
    for path in original_paths.split(':'):
        full_path = os.path.join(base_path, path.lstrip('/'))  # Remove leading '/' to correctly join paths
        os.environ['PATH'] += os.pathsep + full_path

    # Set the XILINX_VIVADO environment variable
    os.environ['XILINX_VITIS'] = os.path.join(base_path, vivado_path.lstrip('/'))
