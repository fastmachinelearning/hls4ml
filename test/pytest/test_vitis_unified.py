import ast
import json
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


@pytest.fixture(scope='module')
def vitis_reference(simple_unet):
    """Plain Vitis backend build of simple_unet, compiled once and shared as the numeric reference."""
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    config['Model']['Strategy'] = 'latency'
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(test_root_path / 'hls4mlprj_test_vitis_unified_vitis_reference'),
        backend='Vitis',
        io_type='io_stream',
        part='xczu9eg-ffvb1156-2-e',
        clock_period=10,
    )
    hls_model.compile()
    return hls_model


@pytest.mark.parametrize(
    'axi_mode, interface_type',
    [
        ('axi_master', 'float'),
        ('axi_master', 'double'),
        ('axi_stream', 'float'),
        ('axi_stream', 'double'),
    ],
)
@pytest.mark.parametrize('np_dtype', [np.float32, np.float64])
def test_predict_any_numpy_dtype(test_case_id, simple_unet, vitis_reference, axi_mode, interface_type, np_dtype):
    # Draw the values as float32 first so the same numbers are exactly representable in both dtypes.
    X_input = np.random.rand(10, 4, 4, 1).astype(np.float32).astype(np_dtype)

    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    config['Model']['Strategy'] = 'latency'
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(test_root_path / test_case_id),
        **_vitis_unified_convert_kwargs('io_stream', axi_mode, input_type=interface_type, output_type=interface_type),
    )
    hls_model.compile()

    prediction = hls_model.predict(X_input)
    reference = vitis_reference.predict(X_input)

    assert np.any(prediction != 0), 'predict() returned all zeros, so the bridge entry point for this dtype is empty'
    np.testing.assert_array_equal(prediction, reference)


@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_bram_weights_rejected_at_conversion(test_case_id, simple_unet, axi_mode):
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    config['Model']['Strategy'] = 'Resource'
    config['Model']['BramFactor'] = 10
    with pytest.raises(Exception, match='BramFactor weights'):
        hls4ml.converters.convert_from_keras_model(
            simple_unet,
            hls_config=config,
            output_dir=str(test_root_path / test_case_id),
            **_vitis_unified_convert_kwargs('io_stream', axi_mode),
        )


@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_writer_options_forwarded(test_case_id, simple_unet, axi_mode):
    output_dir = test_root_path / test_case_id
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(output_dir),
        **_vitis_unified_convert_kwargs('io_stream', axi_mode, namespace='nsone', write_tar=True),
    )
    hls_model.compile()

    header = (output_dir / 'firmware' / 'max_length_project.h').read_text()
    assert 'namespace nsone' in header
    assert output_dir.with_name(output_dir.name + '.tar.gz').exists()

    X_input = np.random.rand(2, 4, 4, 1).astype(np.float32)
    assert np.any(hls_model.predict(X_input) != 0)


FAKE_CSYNTH_XML = """<profile>
<UserAssignments>
<TargetClockPeriod>10.00</TargetClockPeriod>
</UserAssignments>
<PerformanceEstimates>
<SummaryOfTimingAnalysis>
<EstimatedClockPeriod>8.750</EstimatedClockPeriod>
</SummaryOfTimingAnalysis>
<SummaryOfOverallLatency>
<Best-caseLatency>undef</Best-caseLatency>
<Worst-caseLatency>undef</Worst-caseLatency>
<Interval-min>undef</Interval-min>
<Interval-max>undef</Interval-max>
</SummaryOfOverallLatency>
</PerformanceEstimates>
<AreaEstimates>
<Resources>
<BRAM_18K>4</BRAM_18K>
<DSP>171</DSP>
<FF>9744</FF>
<LUT>19834</LUT>
<URAM>0</URAM>
</Resources>
<AvailableResources>
<BRAM_18K>1824</BRAM_18K>
<DSP>2520</DSP>
<FF>548160</FF>
<LUT>274080</LUT>
<URAM>0</URAM>
</AvailableResources>
</AreaEstimates>
</profile>
"""

FAKE_COSIM_RPT = """+--------+--------+-----+-----+-----+-----+-----+-----+
|  RTL   | Status | min | avg | max | min | avg | max |
+--------+--------+-----+-----+-----+-----+-----+-----+
|    VHDL|      NA|   NA|   NA|   NA|   NA|   NA|   NA|
| Verilog|    Pass|  248|  251|  278|  248|  251|  278|
+--------+--------+-----+-----+-----+-----+-----+-----+
"""


@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_build_returns_report_and_reset(test_case_id, simple_unet, axi_mode):
    output_dir = test_root_path / test_case_id
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(output_dir),
        **_vitis_unified_convert_kwargs('io_stream', axi_mode),
    )
    hls_model.write()

    hls_prj = output_dir / 'vitis_workspace' / 'max_length_project' / 'vitis_unified_project' / 'hls'
    (hls_prj / 'syn' / 'report').mkdir(parents=True, exist_ok=True)
    (hls_prj / 'sim' / 'report').mkdir(parents=True, exist_ok=True)
    (hls_prj / 'syn' / 'report' / f'max_length_project_{axi_mode}_csynth.xml').write_text(FAKE_CSYNTH_XML)
    (hls_prj / 'sim' / 'report' / f'max_length_project_{axi_mode}_cosim.rpt').write_text(FAKE_COSIM_RPT)

    with pytest.warns(UserWarning, match='vsynth'):
        report = hls_model.build(vsynth=True)
    assert report['CSynthesisReport']['LUT'] == '19834'
    assert report['CSynthesisReport']['AvailableDSP'] == '2520'
    assert report['CosimReport']['Status'] == 'Pass'
    assert report['CosimReport']['LatencyMax'] == '278'

    hls_model.build(reset=True)
    assert not hls_prj.parent.exists()


@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_config_files_resolved_at_write(test_case_id, simple_unet, axi_mode):
    output_dir = test_root_path / test_case_id
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(output_dir),
        **_vitis_unified_convert_kwargs('io_stream', axi_mode),
    )
    hls_model.write()

    cfgs = {}
    for name in ['csim', 'cosim', 'cosim_fifo_sizing']:
        cfgs[name] = (output_dir / f'hls_kernel_config_{name}.cfg').read_text()
        assert '{' not in cfgs[name]
        assert not any(line.startswith('#') for line in cfgs[name].splitlines())
    assert 'RTL_SIM' not in cfgs['csim'] and 'enable_fifo_sizing' not in cfgs['csim']
    assert '-DRTL_SIM' in cfgs['cosim'] and 'cosim.enable_fifo_sizing=false' in cfgs['cosim']
    assert 'cosim.enable_fifo_sizing=true' in cfgs['cosim_fifo_sizing']

    comp = json.loads((output_dir / 'vitis_workspace' / 'max_length_project' / 'vitis-comp.json').read_text())
    assert all(os.path.isfile(path) for path in comp['configuration']['configFiles'])


# review U12
@pytest.mark.xfail(strict=True, reason='test bench and cfg files still use the myproject name')
@pytest.mark.parametrize('axi_mode', ['axi_stream', 'axi_master'])
def test_custom_project_name(test_case_id, simple_unet, axi_mode):
    output_dir = test_root_path / test_case_id
    config = hls4ml.utils.config_from_keras_model(simple_unet, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        simple_unet,
        hls_config=config,
        output_dir=str(output_dir),
        **_vitis_unified_convert_kwargs('io_stream', axi_mode, project_name='custom'),
    )
    hls_model.write()

    leftovers = []
    for path in output_dir.rglob('*'):
        if not path.is_file() or 'nnet_utils' in path.parts:
            continue
        if 'myproject' in path.name or 'myproject' in path.read_text(errors='ignore'):
            leftovers.append(str(path.relative_to(output_dir)))
    assert leftovers == []


@pytest.mark.parametrize(
    'model_name, bad_kwargs, match',
    [
        ('simple_unet', {'board': 'pynq-z2'}, '(?i)board'),
        ('simple_unet', {'input_type': 'float', 'output_type': 'double'}, '(?i)type'),
        ('multi_io_net', {'axi_mode': 'axi_stream'}, '(?i)axi_stream'),
    ],
    ids=['unknown_board', 'mismatched_types', 'multi_input_axi_stream'],
)
def test_invalid_config_rejected_at_conversion(request, test_case_id, model_name, bad_kwargs, match):
    model = request.getfixturevalue(model_name)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    kwargs = {
        'backend': 'VitisUnified',
        'io_type': 'io_stream',
        'board': 'zcu102',
        'clock_period': 10,
        'axi_mode': 'axi_master',
        **bad_kwargs,
    }
    with pytest.raises(Exception, match=match) as excinfo:
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=str(test_root_path / test_case_id), **kwargs
        )
    assert not isinstance(excinfo.value, AssertionError)


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
@pytest.mark.skipif(
    os.getenv('RUN_VITIS_UNIFIED_BITSTREAM', 'false').lower() not in ('1', 'true'),
    reason='Set RUN_VITIS_UNIFIED_BITSTREAM=true to run bitstream tests',
)
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
