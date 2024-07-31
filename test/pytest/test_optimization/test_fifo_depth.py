import json
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.models import Sequential
import re

import hls4ml

test_root_path = Path(__file__).parent

# backends = ['Vivado', 'Vitis']
io_type_options = ['io_stream', 'io_parallel']
backend_options = ['Vitis']

import os

os.environ['XILINX_VITIS'] = "/opt/Xilinx/Vitis_HLS/2023.2/"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']

def parse_cosim_report(project_path):
    prj_dir = None
    top_func_name = None

    project_path = project_path + '/project.tcl'

    with open(project_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'

    sln_dir = project_path + '/' + prj_dir
    cosim_file_path = sln_dir + f'/sim/report/{top_func_name}_cosim.rpt'

    if os.path.isfile(cosim_file_path):
        return cosim_file_path
    else:
        raise FileNotFoundError("Co-simulation report not found.")

# @pytest.mark.skip(reason='Skipping synthesis tests for now')
def fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type):

    # build a keras model
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
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<8, 4>')
    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=profiling_fifo_depth)

    output_dir = str(test_root_path / f'hls4mlprj_fifo_depth_optimization_backend_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type=io_type, hls_config=config, output_dir=output_dir, backend=backend
    )

    # build the new project with optimized depths
    hls_model.build(reset=False, csim=False, synth=True, cosim=True)
    # hls4ml.report.read_vivado_report(output_dir)

    # checks if the fifo depths decreased
    fifo_depths = {}
    with open(model.config.get_output_dir() + "/fifo_depths.json", "w") as fifo_depths_file:
        fifo_depths = json.load(fifo_depths_file)

    fifo_depths_descreased = True
    for fifo_name in fifo_depths.keys():
        if fifo_depths[fifo_name]['optimized'] >= fifo_depths[fifo_name]['initial']:
            fifo_depths_descreased = False

    # checks that cosimulation ran succesfully without detecting deadlocks
    cosim_report_path = parse_cosim_report(model.config.get_output_dir())

    with open(cosim_report_path) as cosim_report_file:
        cosim_succesful = any(line.strip() == "Pass" for line in cosim_report_file)


    # np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)
    assert cosim_succesful and fifo_depths_descreased

@pytest.mark.parametrize('backend', backend_options)
def test_fifo_depth(backend):
    profiling_fifo_depth = -2
    io_type = 'io_stream'
    value_error_expected_message = "The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer."
    with pytest.raises(ValueError, match=re.escape(value_error_expected_message)):
        fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)
        
    profiling_fifo_depth = "aaa"
    with pytest.raises(ValueError, match=re.escape(value_error_expected_message)):
        fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)
        
    profiling_fifo_depth = 200_000
    io_type = 'io_parallel'
    runtime_error_expected_message = "To use this optimization you have to set `IOType` field to `io_stream` in the HLS config."
    with pytest.raises(RuntimeError, match=re.escape(runtime_error_expected_message)):
            fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)
            
    # profiling_fifo_depth = "asdada"
    # io_type = 'io_stream'
    # with pytest.raises(ValueError, match="The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer"):
    #     fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)
