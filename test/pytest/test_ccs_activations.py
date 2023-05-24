# This is a copy of test_activations.py that is specialized to test the Catapult backend.
# It covers model generation, C++ generation, C++ compilation with SCVerify/OSCI, Catapult HLS and SCVerify RTL sim

from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import ELU, Activation, Input, LeakyReLU, ReLU, ThresholdedReLU
from tensorflow.keras.models import Model
import re
import os

import hls4ml

test_root_path = Path(__file__).parent

# Variable 'name' is simply used as an identifier for the activation


@pytest.mark.parametrize('backend', ['Catapult'])
@pytest.mark.parametrize('shape, io_type', [((8,), 'io_parallel')])
@pytest.mark.parametrize(
    'activation, name',
    [
        (LeakyReLU(alpha=1.5), 'leaky_relu'),
    ],
)
def test_ccs_activations(backend, activation, name, shape, io_type):
    # Subtract 0.5 to include negative values
    X = np.random.rand(1000, *shape) - 0.5

    input = Input(shape=shape)
    activation = activation(input)
    keras_model = Model(inputs=input, outputs=activation)

    hls_config = hls4ml.utils.config_from_keras_model(keras_model)
    output_dir = str(test_root_path / 'hls4mlprj_activations_{}_{}_{}_{}').format(backend, io_type, re.sub('[)(, ]', '_',str(shape)), name)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_config, io_type=io_type, output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    # Run through 'go compile', run SCVerify OSCI, run HLS synth, run SCVerify RTL, do not run Vivado synth
    hls_model.build(csim=True,synth=True,cosim=True,vsynth=False)

    # If Catapult ran, then there should be a catapult.log file
    cat_log = os.path.join(output_dir, "catapult.log")   
    assert os.path.exists(cat_log) == True

    # If Catapult ran, then there should be a project file
    cat_prj = os.path.join(output_dir, "myproject_prj.ccs")   
    assert os.path.exists(cat_prj) == True
  
    # If Catapult ran, then there should be an RTL netlist
    rtl_v = os.path.join(output_dir, "myproject_prj/myproject.v1/rtl.v")   
    assert os.path.exists(rtl_v) == True

    with open(cat_log) as myfile:
        contents = myfile.read()
        assert 'Simulation FAILED' not in contents
        assert 'Simulation PASSED' in contents

