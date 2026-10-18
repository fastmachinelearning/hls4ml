import inspect
import os
import re
import shutil
from pathlib import Path

import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml
from hls4ml.report.vivado_report import _parse_csynth_xml, _parse_vivado_utilization

test_root_path = Path(__file__).parent


def copy_vivado_report(output_dir, test_report_dir):
    # copy pregenerated Vivado reports
    os.makedirs(f'{output_dir}/myproject_prj/solution1/syn/report', exist_ok=True)
    shutil.copy(test_report_dir / 'vivado_hls.app', f'{output_dir}/myproject_prj/vivado_hls.app')
    shutil.copy(
        test_report_dir / 'myproject_csynth.rpt', f'{output_dir}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'
    )
    shutil.copy(
        test_report_dir / 'myproject_csynth.xml', f'{output_dir}/myproject_prj/solution1/syn/report/myproject_csynth.xml'
    )
    shutil.copy(test_report_dir / 'vivado_synth.rpt', f'{output_dir}/vivado_synth.rpt')

    return


def copy_altera_report(output_dir, test_report_dir):
    # copy pregenerated Altera reports
    json_dir = f'{output_dir}/build/myproject.fpga.prj/reports/resources/json'
    os.makedirs(json_dir, exist_ok=True)
    shutil.copy(test_report_dir / 'quartus.ndjson', f'{json_dir}/quartus.ndjson')
    shutil.copy(test_report_dir / 'summary.ndjson', f'{json_dir}/summary.ndjson')
    shutil.copy(test_report_dir / 'loop_attr.ndjson', f'{json_dir}/loop_attr.ndjson')

    return


@pytest.fixture
def backend_configs():
    """Returns configuration settings for different backends."""
    config_dict = {
        'Vivado': {
            'backend': 'Vivado',
            'part': 'xc7z020clg400-1',
            'build': {'synth': True, 'vsynth': True},
            'copy_func': copy_vivado_report,
            'parse_func': hls4ml.report.parse_vivado_report,
            'print_func': hls4ml.report.print_vivado_report,
            'expected_outcome': '\n'
            + '======================================================\n'
            + '== C Synthesis report\n'
            + '======================================================\n\n'
            + ' - Performance estimates:\n'
            + '    Best-case latency:      10 (50.0 ns)\n'
            + '    Worst-case latency:     10 (50.0 ns)\n'
            + '    Interval Min:           8\n'
            + '    Interval Max:           8\n'
            + '    Estimated Clock Period: 4.049\n\n'
            + ' - Resource estimates:\n'
            + '    BRAM_18K: 0 / 280 (0.0%)\n'
            + '    DSP:      73 / 220 (33.2%)\n'
            + '    FF:       7969 / 106400 (7.5%)\n'
            + '    LUT:      2532 / 53200 (4.8%)\n'
            + '    URAM:     N/A\n\n'
            + '======================================================\n'
            + '== Vivado Synthesis report\n'
            + '======================================================\n\n'
            + ' - Resource utilization:\n'
            + '    BRAM_18K: 0\n'
            + '    DSP48E:   66\n'
            + '    FF:       2428\n'
            + '    LUT:      1526\n'
            + '    URAM:     N/A\n\n',
        },
        'Altera': {
            'backend': 'Altera',
            'part': 'Agilex7',
            'build': {'build_type': 'fpga'},
            'copy_func': copy_altera_report,
            'parse_func': hls4ml.report.parse_altera_report,
            'print_func': hls4ml.report.print_altera_report,
            'expected_outcome': '\n'
            + '==================================================\n'
            + '== FPGA Hardware Synthesis\n'
            + '==================================================\n\n'
            + ' - Performance estimates:\n'
            + '    Minimum Frequency (HLS):   480.0            \n'
            + '    Worst-case latency (HLS):  200.0            \n'
            + '    Max II (HLS):              1                \n'
            + '    Maximum Frequency:         597.73           \n\n'
            + ' - Resource estimates:\n'
            + '    :       Quartus Synthesis HLS Estimation    Available        \n'
            + '    ALUTs:  4181 (0.4%)       2462 (0.3%)       974400           \n'
            + '    FFs:    16419 (0.8%)      7938 (0.4%)       1948800          \n'
            + '    DSPs:   40 (0.9%)         0 (< 0.1%)        4510             \n'
            + '    RAMs:   36 (0.5%)         77 (1.1%)         7110             \n'
            + '    MLABs:  52 (0.2%)         92 (0.4%)         24360            \n'
            + '    ALMs:   4520.0            N/A               N/A              \n\n',
        },
    }

    return config_dict


@pytest.fixture
def hls_model_setup(request, test_case_id, backend_configs, tmp_path):
    """Fixture to create, write, and copy the report files of the HLS model
    for a given backend."""
    backend_config = backend_configs[request.param]

    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')

    output_dir = str(tmp_path / test_case_id)
    test_report_dir = test_root_path / f'test_report/{backend_config["backend"]}'

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        io_type='io_stream',
        hls_config=config,
        output_dir=output_dir,
        part=backend_config['part'],
        backend=backend_config['backend'],
    )
    hls_model.write()

    # to actually generate the reports (using Vivado, Vitis, or Altera HLS IP Gen)
    # hls_model.build(**(backend_config['build']))

    backend_config['copy_func'](output_dir, test_report_dir)

    yield output_dir, backend_config


@pytest.mark.parametrize('hls_model_setup', ['Vivado', 'Altera'], indirect=True)
def test_report(hls_model_setup, capsys):
    """Tests that the report parsing and printing functions work for different backends."""
    output_dir, backend_config = hls_model_setup

    report = backend_config['parse_func'](output_dir)

    capsys.readouterr()  # capture to clear
    backend_config['print_func'](report)
    captured = capsys.readouterr()  # capture again to test

    assert captured.out == backend_config['expected_outcome']


def test_altera_report_multiple_targets(tmp_path):
    """Each build target found in the directory gets its own entry in the report."""
    test_report_dir = test_root_path / 'test_report/Altera'
    for target in ['report', 'fpga']:
        json_dir = tmp_path / f'build/myproject.{target}.prj/reports/resources/json'
        json_dir.mkdir(parents=True)
        for json_file in ['quartus.ndjson', 'summary.ndjson', 'loop_attr.ndjson']:
            shutil.copy(test_report_dir / json_file, json_dir / json_file)

    report = hls4ml.report.parse_altera_report(str(tmp_path))

    assert set(report.keys()) == {'report', 'fpga'}
    assert 'Quartus' in report['fpga']
    assert 'Quartus' not in report['report']


vivado_report_dir = test_root_path / 'test_report/Vivado'


def make_vivado_project_dir(output_dir, project_name, backend):
    """Create the files parse_vivado_report needs to find an HLS project, without writing a model."""
    os.makedirs(f'{output_dir}/{project_name}_prj/solution1', exist_ok=True)
    shutil.copy(vivado_report_dir / 'vivado_hls.app', f'{output_dir}/{project_name}_prj/vivado_hls.app')
    with open(f'{output_dir}/project.tcl', 'w') as f:
        f.write(f'set project_name "{project_name}"\nset backend "{backend}"\n')


# The utilization reports other than vivado_synth.rpt are of a small Verilog module with one block RAM of 36 Kb and one
# DSP, cut to the sections that are read.
@pytest.mark.parametrize(
    'report_file, expected',
    [
        # 7-series after synthesis: 'Slice' names, and a '*' after the name of the LUT row
        ('vivado_synth.rpt', {'LUT': '1526', 'FF': '2428', 'BRAM_18K': '0', 'DSP48E': '66'}),
        # UltraScale after synthesis: 'CLB' names, and no URAM on the device
        ('utilization_ultrascale.rpt', {'LUT': '92', 'FF': '104', 'BRAM_18K': '2', 'DSP48E': '1'}),
        # UltraScale+ after place and route: the table for each SLR repeats the names with other values
        ('utilization_ultrascale_plus.rpt', {'LUT': '91', 'FF': '104', 'BRAM_18K': '2', 'URAM': '0', 'DSP48E': '1'}),
        # Versal after place and route: 'Registers' and 'DSP Slices', then 'CLB Registers' in a later table
        ('utilization_versal.rpt', {'LUT': '56', 'FF': '56', 'BRAM_18K': '2', 'URAM': '0', 'DSP48E': '1'}),
    ],
)
def test_vivado_utilization(report_file, expected):
    """The resources are read from the utilization reports of every device family."""
    assert _parse_vivado_utilization(vivado_report_dir / report_file) == expected


def test_vivado_impl_report(tmp_path, capsys):
    """The resources, slack and power after place and route are read and printed."""
    make_vivado_project_dir(tmp_path, 'myproject', 'vivado')
    shutil.copy(vivado_report_dir / 'utilization_ultrascale_plus.rpt', tmp_path / 'vivado_impl.rpt')
    shutil.copy(vivado_report_dir / 'timing_summary.rpt', tmp_path / 'vivado_impl_timing.rpt')
    shutil.copy(vivado_report_dir / 'power.rpt', tmp_path / 'vivado_impl_power.rpt')

    report = hls4ml.report.parse_vivado_report(str(tmp_path))
    capsys.readouterr()
    hls4ml.report.print_vivado_report(report)

    assert capsys.readouterr().out == (
        '\n'
        + '======================================================\n'
        + '== Vivado Place and Route report\n'
        + '======================================================\n\n'
        + ' - Resource utilization:\n'
        + '    BRAM_18K: 2\n'
        + '    DSP48E:   1\n'
        + '    FF:       104\n'
        + '    LUT:      91\n'
        + '    URAM:     0\n\n'
        + ' - Timing:\n'
        + '    Worst Negative Slack (WNS): 1.452\n'
        + '    Total Negative Slack (TNS): 0.0\n'
        + '    Worst Hold Slack (WHS):     0.055\n'
        + '    Total Hold Slack (THS):     0.0\n\n'
        + ' - Power:\n'
        + '    Total On-Chip Power (W): 2.95\n'
        + '    Dynamic (W):             0.012\n'
        + '    Device Static (W):       2.938\n\n'
    )


def test_vivado_accelerator_reports(tmp_path, capsys):
    """The system reports of VivadoAccelerator are read without co-simulation and for a project name with '_'."""
    make_vivado_project_dir(tmp_path, 'my_project', 'vivadoaccelerator')
    shutil.copy(vivado_report_dir / 'utilization_hierarchical.rpt', tmp_path / 'util.rpt')
    timing_dir = tmp_path / 'my_project_vivado_accelerator/project_1.runs/impl_1'
    timing_dir.mkdir(parents=True)
    shutil.copy(vivado_report_dir / 'timing_summary.rpt', timing_dir / 'design_1_wrapper_timing_summary_routed.rpt')

    report = hls4ml.report.parse_vivado_report(str(tmp_path))

    assert report['ImplementationReport']['TotLUTs'] == 92
    assert report['TimingReport']['WNS'] == 1.452
    capsys.readouterr()
    hls4ml.report.print_vivado_report(report)
    assert 'Total LUTs: 92 (0.17%)' in capsys.readouterr().out


def test_vitis_unified_bitfile_report(tmp_path):
    """After a bitfile build, the resources of the kernel and the slack and power of the linked design are read."""
    with open(tmp_path / 'hls4ml_config.yml', 'w') as f:
        f.write('ProjectName: myproject\nVitisUnifiedConfig:\n  axi_mode: axi_stream\n')
    report_dir = tmp_path / 'final_reports'
    report_dir.mkdir()
    shutil.copy(
        test_root_path / 'test_report/VitisUnified/kernel_utilization.rpt', report_dir / 'impl_1_kernel_util_routed.rpt'
    )
    shutil.copy(
        vivado_report_dir / 'timing_summary.rpt', report_dir / 'impl_1_vitis_design_wrapper_timing_summary_routed.rpt'
    )
    shutil.copy(vivado_report_dir / 'power.rpt', report_dir / 'impl_1_power_routed.rpt')

    report = hls4ml.report.parse_vitis_unified_report(str(tmp_path))

    assert {key: report['BitfileReport'][key] for key in ['LUT', 'FF', 'BRAM_18K', 'DSP48E', 'WNS', 'TotalPower']} == {
        'LUT': '6869',
        'FF': '3769',
        'BRAM_18K': '16',
        'DSP48E': '133',
        'WNS': 1.452,
        'TotalPower': 2.95,
    }


def test_csynth_worst_case_latency(capsys):
    """The worst-case latency is printed from WorstLatency."""
    csynth_report = _parse_csynth_xml(vivado_report_dir / 'myproject_csynth.xml')
    csynth_report['WorstLatency'] = '12'

    hls4ml.report.print_vivado_report({'CSynthesisReport': csynth_report})

    assert 'Worst-case latency:     12 (60.0 ns)' in capsys.readouterr().out


def test_aggregate_graph_reports():
    """The DSPs of every subgraph are summed, and the clock period comes from C synthesis, with VivadoSynthReport."""

    def subgraph_report(dsp, clock_period):
        return {
            'CSynthesisReport': {'EstimatedClockPeriod': clock_period, 'DSP': '0'},
            'VivadoSynthReport': {'DSP48E': dsp},
        }

    report = hls4ml.report.aggregate_graph_reports(
        {'graph1': subgraph_report('3', '4.1'), 'graph2': subgraph_report('5', '4.5')}
    )

    assert report['StitchedDesignReport']['DSP'] == '8.0'
    assert report['StitchedDesignReport']['EstimatedClockPeriod'] == '4.500'


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_build_options_have_defaults(backend):
    """build_prj.tcl sets a default for every option of build(), so a build_opt.tcl without some of them works."""
    with open(Path(hls4ml.__file__).parent / f'templates/{backend.lower()}/build_prj.tcl') as f:
        defaults = re.search(r'array set opt \{(.*?)\}', f.read(), re.DOTALL).group(1).split()[::2]
    options = inspect.signature(hls4ml.backends.get_backend(backend).build).parameters

    assert set(options) - {'model', 'log_to_stdout'} <= set(defaults)
