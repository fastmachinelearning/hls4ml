import os
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent
# 1. For who installed the xilinx tool chain in /opt
# os.environ['XILINX_VITIS'] = '/opt/Xilinx/Vitis/2023.2'

# 2. For who installed the xilinx tool chain in /tools (both vitis and vivado are mandatory)
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2023.2'
os.environ['XILINX_VIVADO'] = '/tools/Xilinx/Vivado/2023.2'

os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']


# @pytest.fixture(scope='module')
def simple_unet():
    """Simple U-Net model for Vitis Unified tests."""
    inputs = Input((4, 4, 1))
    c1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    bn = Conv2D(4, (3, 3), activation='relu', padding='same')(p1)
    u1 = UpSampling2D((2, 2))(bn)
    concat1 = Concatenate()([u1, c1])
    c2 = Conv2D(2, (3, 3), activation='relu', padding='same')(concat1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c2)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


part_map = {'zcu102': 'xczu9eg-ffvb1156-2-e', 'kv260': 'xck26-sfvc784-2LV-c'}


def _vitis_unified_convert_kwargs(io_type, axi_mode, board='zcu102', **extra):
    """Shared backend kwargs for VitisUnified conversion.
    Platform is resolved from supported_boards.json by board + axi_mode.
    """
    part = part_map[board]
    return {
        'backend': 'VitisUnified',
        'io_type': io_type,
        'board': board,
        'part': part,
        'clock_period': '10ns',
        'input_type': 'float',
        'output_type': 'float',
        'axi_mode': axi_mode,
        **extra,
    }


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_backend_predict(test_case_id, simple_unet, io_type, strategy, granularity, batch_size, axi_mode):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    output_dir_unified = str(test_root_path / test_case_id)
    output_dir_vitis = str(test_root_path / (test_case_id + '_vitis_ref'))

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir_unified,
        **_vitis_unified_convert_kwargs(io_type, axi_mode),
    )
    vitis_unified_model.compile()
    vitis_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir_vitis,
        backend='Vitis',
        io_type=io_type,
        part='xczu9eg-ffvb1156-2-e',
        clock_period='10ns',
    )
    vitis_model.compile()

    hls_unified_prediction = vitis_unified_model.predict(X_input)
    hls_vitis_prediction = vitis_model.predict(X_input)

    np.testing.assert_array_equal(hls_unified_prediction, hls_vitis_prediction)


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_cosimulation(test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)
    np.save(tmp_path / 'input.npy', X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode),
    )
    vitis_unified_model.compile()
    y_pred = vitis_unified_model.predict(X_input)
    np.save(tmp_path / 'output.npy', y_pred)

    input_data_tb = str(tmp_path / 'input.npy')
    output_data_tb = str(tmp_path / 'output.npy')

    vitis_unified_model_cosim = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, input_data_tb=input_data_tb, output_data_tb=output_data_tb),
    )
    vitis_unified_model_cosim.compile()
    vitis_unified_model_cosim.build(synth=True, cosim=True, log_to_stdout=True)

    bridge_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'tb_output_predictions.dat'))
    cosim_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'rtl_cosim_results.log'))
    assert np.allclose(bridge_result, cosim_result, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_csim_simulation(test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)
    np.save(tmp_path / 'input.npy', X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode),
    )
    vitis_unified_model.compile()
    y_pred = vitis_unified_model.predict(X_input)
    np.save(tmp_path / 'output.npy', y_pred)

    input_data_tb = str(tmp_path / 'input.npy')
    output_data_tb = str(tmp_path / 'output.npy')

    vitis_unified_model_csim = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, input_data_tb=input_data_tb, output_data_tb=output_data_tb),
    )
    vitis_unified_model_csim.compile()
    vitis_unified_model_csim.build(synth=True, csim=True, log_to_stdout=True)

    bridge_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'tb_output_predictions.dat'))
    csim_result = np.loadtxt(os.path.join(output_dir, 'tb_data', 'csim_results.log'))
    assert np.allclose(bridge_result, csim_result, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_fifo_depth(test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)
    np.save(tmp_path / 'input.npy', X_input)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    config['Flows'] = ['vitisunified:fifo_depth_optimization']
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode),
    )
    vitis_unified_model.compile()
    y_pred = vitis_unified_model.predict(X_input)
    np.save(tmp_path / 'output.npy', y_pred)

    input_data_tb = str(tmp_path / 'input.npy')
    output_data_tb = str(tmp_path / 'output.npy')

    vitis_unified_model_fifo = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, input_data_tb=input_data_tb, output_data_tb=output_data_tb),
    )
    vitis_unified_model_fifo.compile()

    fifodepth_result_path = os.path.join(output_dir, 'fifo_depths.json')
    assert os.path.exists(fifodepth_result_path)


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
# @pytest.mark.parametrize('board', ['zcu102', 'kv260'])
@pytest.mark.parametrize('board', ['kv260'])
def test_gen_unified(test_case_id, simple_unet, io_type, strategy, granularity, batch_size, axi_mode, board):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    test_case_id = test_case_id
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, board),
    )
    vitis_unified_model.compile()
    # predict and save for hardware comparison purpose
    y_pred = vitis_unified_model.predict(X_input)
    np.save(os.path.join(output_dir, 'x_input.npy'), X_input)
    np.save(os.path.join(output_dir, 'y_pred_sw.npy'), y_pred)
    vitis_unified_model.build(synth=True, bitfile=True, log_to_stdout=True)

    export_dir = os.path.join(output_dir, 'export')
    driver_file = 'axi_stream_driver.py' if axi_mode == 'axi_stream' else 'axi_master_driver.py'
    expected_files = {driver_file, 'system.bit', 'system.hwh', 'driver.py'}
    exported_files = set(os.listdir(export_dir))
    assert expected_files.issubset(exported_files), f'Missing files in export: {expected_files - exported_files}'
    final_reports_dir = os.path.join(output_dir, 'final_reports')
    assert os.path.isdir(final_reports_dir), f'final_reports directory does not exist: {final_reports_dir}'
    rpt_files = [f for f in os.listdir(final_reports_dir) if f.endswith('.rpt')]
    assert len(rpt_files) > 0, f'No .rpt files found in final_reports directory: {final_reports_dir}'


test_gen_unified('axi_stream_debug', simple_unet(), 'io_stream', 'latency', 'name', 10, 'axi_stream', 'kv260')
