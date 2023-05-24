# This file mimics the testing done in test_keras_api.py except that the intent is to not check the model but
# rather to run a single layer network through HLS4ML.
# It covers model generation, C++ generation, C++ compilation with SCVerify/OSCI, Catapult HLS and SCVerify RTL sim

import math
from pathlib import Path
import re
import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    Conv1D,
    Conv2D,
    Dense,
    LeakyReLU,
    MaxPooling1D,
    MaxPooling2D,
    PReLU,
)

import hls4ml

test_root_path = Path(__file__).parent

@pytest.mark.parametrize('backend', ['Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def test_ccs_dense(backend, io_type, strategy):
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

    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / f'hls4mlprj_ccs_dense_{backend}_{io_type}_{strategy}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
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

