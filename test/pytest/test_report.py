import os
import shutil
from pathlib import Path

import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('backend', ['Vivado'])
def test_report(backend, capsys):
    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')

    output_dir = str(test_root_path / f'hls4mlprj_report_{backend}')
    test_report_dir = test_root_path / 'test_report'

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type='io_stream', hls_config=config, output_dir=output_dir, part='xc7z020clg400-1', backend=backend
    )
    hls_model.write()

    # to actually generate the reports (using Vivado 2020.1)
    # hls_model.build(synth=True, vsynth=True)

    # copy pregenerated reports
    os.makedirs(f'{output_dir}/myproject_prj/solution1/syn/report', exist_ok=True)
    shutil.copy(test_report_dir / 'vivado_hls.app', f'{output_dir}/myproject_prj/vivado_hls.app')
    shutil.copy(
        test_report_dir / 'myproject_csynth.rpt', f'{output_dir}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'
    )
    shutil.copy(
        test_report_dir / 'myproject_csynth.xml', f'{output_dir}/myproject_prj/solution1/syn/report/myproject_csynth.xml'
    )
    shutil.copy(test_report_dir / 'vivado_synth.rpt', f'{output_dir}/vivado_synth.rpt')

    report = hls4ml.report.parse_vivado_report(output_dir)  # or report = hls_model.build(...)

    capsys.readouterr()  # capture to clear
    hls4ml.report.print_vivado_report(report)
    captured = capsys.readouterr()  # capture again to test

    assert (
        captured.out
        == '\n'
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
        + '    URAM:     N/A\n\n'
    )
