import ast
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


def require_synthesis(synthesis_config):
    if not synthesis_config['run_synthesis']:
        pytest.skip('Set RUN_SYNTHESIS=true to run synthesis tests')


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def multi_io_net():
    """Two-input / two-output CNN, used to exercise the AXI-master multi-port path.

    Kept deliberately small so the bitstream build stays comparable to simple_unet;
    the point is the port count, not the model.
    """
    in_a = Input((4, 4, 1), name='in_a')
    in_b = Input((4, 4, 1), name='in_b')
    conv_a = Conv2D(2, (3, 3), activation='relu', padding='same', name='conv_a')(in_a)
    conv_b = Conv2D(2, (3, 3), activation='relu', padding='same', name='conv_b')(in_b)
    merged = Concatenate(name='merge')([conv_a, conv_b])
    trunk = Conv2D(2, (3, 3), activation='relu', padding='same', name='trunk')(merged)
    out_a = Conv2D(1, (1, 1), activation='sigmoid', name='out_a')(trunk)
    out_b = Conv2D(1, (1, 1), activation='sigmoid', name='out_b')(trunk)
    model = Model([in_a, in_b], [out_a, out_b])
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
        'clock_period': 10,
        'input_type': 'float',
        'output_type': 'float',
        'axi_mode': axi_mode,
        'project_name': 'max_length_project',  # 18 chars → decl = 63 chars (limit is 64)
        **extra,
    }


def _driver_port_counts(driver_path):
    """Length of each per-port list the writer emits into the generated AXI-master driver."""
    wanted = {'INP_PORT_NAMEs', 'REG_ADDR_INP_PTRs', 'OUT_PORT_NAMEs', 'REG_ADDR_OUT_PTRs'}
    with open(driver_path) as f:
        tree = ast.parse(f.read())
    counts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in wanted:
                counts[target.attr] = len(node.value.elts)
    return counts


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
        clock_period=10,
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
def test_cosimulation(
    test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode, synthesis_config
):
    require_synthesis(synthesis_config)
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
def test_csim_simulation(
    test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode, synthesis_config
):
    require_synthesis(synthesis_config)
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
def test_fifo_depth(
    test_case_id, simple_unet, tmp_path, io_type, strategy, granularity, batch_size, axi_mode, synthesis_config
):
    require_synthesis(synthesis_config)
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
# Keep full bitstream generation out of regular CI for now; revisit it as part of PR #1474.
@pytest.mark.skipif(
    os.getenv('RUN_VITIS_UNIFIED_BITSTREAM', 'false').lower() not in ('1', 'true'),
    reason='Set RUN_VITIS_UNIFIED_BITSTREAM=true to run bitstream tests',
)
def test_gen_unified(test_case_id, simple_unet, io_type, strategy, granularity, batch_size, axi_mode, board):
    model = simple_unet
    X_input = np.random.rand(batch_size, 4, 4, 1).astype(np.float32)

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
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
    expected_files = {driver_file, 'system.bit', 'system.hwh'}
    exported_files = set(os.listdir(export_dir))
    assert expected_files.issubset(exported_files), f'Missing files in export: {expected_files - exported_files}'
    final_reports_dir = os.path.join(output_dir, 'final_reports')
    assert os.path.isdir(final_reports_dir), f'final_reports directory does not exist: {final_reports_dir}'
    rpt_files = [f for f in os.listdir(final_reports_dir) if f.endswith('.rpt')]
    assert len(rpt_files) > 0, f'No .rpt files found in final_reports directory: {final_reports_dir}'


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('batch_size', [10])
# axi_stream carries a single DMA stream in each direction, so multi-port requires axi_master
@pytest.mark.parametrize('axi_mode', ['axi_master'])
# @pytest.mark.parametrize('board', ['zcu102', 'kv260'])
@pytest.mark.parametrize('board', ['kv260'])
def test_gen_unified_multi_io(test_case_id, multi_io_net, io_type, strategy, granularity, batch_size, axi_mode, board):
    """Full bitstream generation for a multi-input / multi-output model on AXI-master.

    Mirrors test_gen_unified, plus asserts that the generated driver exposes one
    pointer register per model port rather than collapsing onto port 0.
    """
    model = multi_io_net
    X_inputs = [np.random.rand(batch_size, 4, 4, 1).astype(np.float32) for _ in model.inputs]

    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, board),
    )
    vitis_unified_model.compile()

    export_dir = os.path.join(output_dir, 'export')
    driver_file = 'axi_master_driver.py'
    driver_path = os.path.join(export_dir, driver_file)

    # the driver is emitted at write time, so check the multi-port wiring before
    # spending a full synthesis run on it
    assert os.path.isfile(driver_path), f'driver was not generated: {driver_path}'
    counts = _driver_port_counts(driver_path)
    n_in, n_out = len(model.inputs), len(model.outputs)
    assert counts.get('INP_PORT_NAMEs') == n_in, f'expected {n_in} input port names, got {counts}'
    assert counts.get('REG_ADDR_INP_PTRs') == n_in, f'expected {n_in} input pointer regs, got {counts}'
    assert counts.get('OUT_PORT_NAMEs') == n_out, f'expected {n_out} output port names, got {counts}'
    assert counts.get('REG_ADDR_OUT_PTRs') == n_out, f'expected {n_out} output pointer regs, got {counts}'
    # every pointer register must be distinct, otherwise ports would alias each other
    assert len(set(counts)) == 4, f'missing per-port lists in generated driver: {counts}'

    # predict and save for hardware comparison purpose
    y_pred = vitis_unified_model.predict(X_inputs)
    y_pred = list(y_pred) if isinstance(y_pred, (list, tuple)) else [y_pred]
    assert len(y_pred) == n_out, f'expected {n_out} output arrays from predict, got {len(y_pred)}'
    for idx, x_input in enumerate(X_inputs):
        np.save(os.path.join(output_dir, f'x_input_{idx}.npy'), x_input)
    for idx, y_out in enumerate(y_pred):
        np.save(os.path.join(output_dir, f'y_pred_sw_{idx}.npy'), y_out)

    vitis_unified_model.build(synth=True, bitfile=True, log_to_stdout=True)

    expected_files = {driver_file, 'system.bit', 'system.hwh'}
    exported_files = set(os.listdir(export_dir))
    assert expected_files.issubset(exported_files), f'Missing files in export: {expected_files - exported_files}'
    final_reports_dir = os.path.join(output_dir, 'final_reports')
    assert os.path.isdir(final_reports_dir), f'final_reports directory does not exist: {final_reports_dir}'
    rpt_files = [f for f in os.listdir(final_reports_dir) if f.endswith('.rpt')]
    assert len(rpt_files) > 0, f'No .rpt files found in final_reports directory: {final_reports_dir}'


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_project_name_too_long(test_case_id, simple_unet, io_type, strategy, granularity, axi_mode):
    model = simple_unet
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / test_case_id)

    vitis_unified_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        **_vitis_unified_convert_kwargs(io_type, axi_mode, project_name='name_exceeds_limits'),  # 19 chars → decl = 65 chars
    )
    with pytest.raises(ValueError, match='Project name must not exceed 18 characters'):
        vitis_unified_model.compile()


# test_gen_unified('axi_stream_debug_4', simple_unet(), 'io_stream', 'latency', 'name', 10, 'axi_stream', 'kv260')
