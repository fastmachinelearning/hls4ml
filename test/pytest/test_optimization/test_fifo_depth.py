from pathlib import Path

import json
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
def fifo_depth_optimization_script(backend):

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
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=200_000)

    output_dir = str(test_root_path / f'hls4mlprj_fifo_depth_optimization_backend_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type='io_stream', hls_config=config, output_dir=output_dir, backend=backend
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

@pytest.mark.parametrize('backend', backends)
def test_fifo_depth():
    