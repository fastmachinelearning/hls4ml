import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.models import Sequential

import hls4ml
from hls4ml.backends.vitis.passes.fifo_depth_optimization import override_test_bench

test_root_path = Path(__file__).parent

backend_options = ['Vitis']


def parse_cosim_report(project_path):
    prj_dir = None
    top_func_name = None

    project_tcl_path = project_path + '/project.tcl'

    with open(project_tcl_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'

    cosim_file_path = project_path + '/' + prj_dir + f'/solution1/sim/report/{top_func_name}_cosim.rpt'

    if os.path.isfile(cosim_file_path):
        return cosim_file_path
    else:
        raise FileNotFoundError("Co-simulation report not found.")


def fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type):
    # create a keras model
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

    # execute fifo optimization
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32, 16>')
    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(
        profiling_fifo_depth=profiling_fifo_depth
    )

    output_dir = str(test_root_path / f'hls4mlprj_fifo_depth_optimization_backend_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type=io_type, hls_config=config, output_dir=output_dir, backend=backend
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)

    # force the top-function to execute twice in the cosimulation, to verify no deadlocks occur even
    # when streaming multiple inputs into the network
    override_test_bench(hls_model)

    # build the new project with optimized depths and execute cosimulation to check for deadlocks
    # due to the new FIFO depths
    hls_model.build(reset=False, csim=False, synth=True, cosim=True)

    # checks if the fifo depths decreased
    fifo_depths = {}
    with open(hls_model.config.get_output_dir() + "/fifo_depths.json") as fifo_depths_file:
        fifo_depths = json.load(fifo_depths_file)

    fifo_depths_decreased = all(fifo['optimized'] < fifo['initial'] for fifo in fifo_depths.values())

    # checks that cosimulation ran succesfully without detecting deadlocks
    cosim_report_path = parse_cosim_report(hls_model.config.get_output_dir())

    with open(cosim_report_path) as cosim_report_file:
        cosim_succesful = any("Pass" in line for line in cosim_report_file)

    assert cosim_succesful and fifo_depths_decreased


def expect_exception(error, message, backend, profiling_fifo_depth, io_type):
    with pytest.raises(error, match=re.escape(message)):
        fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)


# test faulty inputs of profiling_fifo_depth to verify that an exception is raised
@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
@pytest.mark.parametrize('profiling_fifo_depth', [-2, "a"])
def test_value_error(backend, profiling_fifo_depth):
    message = "The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer."
    expect_exception(ValueError, message, backend, profiling_fifo_depth, io_type='io_stream')


# test with io_type='io_parallel' to verify that an exception is raised
@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_runtime_error(backend):
    message = "To use this optimization you have to set `IOType` field to `io_stream` in the HLS config."
    expect_exception(RuntimeError, message, backend, profiling_fifo_depth=200_000, io_type='io_parallel')


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution(backend):
    fifo_depth_optimization_script(backend, profiling_fifo_depth=200_000, io_type='io_stream')
