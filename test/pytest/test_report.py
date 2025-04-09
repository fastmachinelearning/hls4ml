import os
import shutil
from pathlib import Path

import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

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


def copy_oneapi_report(output_dir, test_report_dir):
    # copy pregenerated oneAPI reports
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
        'oneAPI': {
            'backend': 'oneAPI',
            'part': 'Agilex7',
            'build': {'build_type': 'fpga'},
            'copy_func': copy_oneapi_report,
            'parse_func': hls4ml.report.parse_oneapi_report,
            'print_func': hls4ml.report.print_oneapi_report,
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
def hls_model_setup(request, backend_configs, tmp_path):
    """Fixture to create, write, and copy the report files of the HLS model
    for a given backend."""
    backend_config = backend_configs[request.param]

    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')

    output_dir = str(tmp_path / f'hls4mlprj_report_{backend_config["backend"]}')
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

    # to actually generate the reports (using Vivado 2020.1 or oneAPI 2025.0)
    # hls_model.build(**(backend_config['build']))

    backend_config["copy_func"](output_dir, test_report_dir)

    yield output_dir, backend_config


@pytest.mark.parametrize("hls_model_setup", ['Vivado', 'oneAPI'], indirect=True)
def test_report(hls_model_setup, capsys):
    """Tests that the report parsing and printing functions work for different backends."""
    output_dir, backend_config = hls_model_setup

    report = backend_config['parse_func'](output_dir)

    capsys.readouterr()  # capture to clear
    backend_config['print_func'](report)
    captured = capsys.readouterr()  # capture again to test

    assert captured.out == backend_config['expected_outcome']
