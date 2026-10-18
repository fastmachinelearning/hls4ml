import glob
import json
import os
import shutil
import subprocess
import tempfile

import yaml

from hls4ml.report.vivado_report import (
    _parse_cosim_rpt,
    _parse_csynth_xml,
    _parse_vivado_power,
    _parse_vivado_synth_reports,
    _parse_vivado_timing_summary,
)


class _ConfigLoader(yaml.SafeLoader):
    pass


# hls4ml_config.yml stores the source model with a custom tag such as !keras_model, which is not needed here
_ConfigLoader.add_multi_constructor('!', lambda loader, suffix, node: None)


def read_kernel_clock(xclbin):
    """Kernel clock in MHz stored in an .xclbin (DATA_CLK), or None if it cannot be read."""
    if not os.path.isfile(xclbin) or shutil.which('xclbinutil') is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'clock.json')
        cmd = ['xclbinutil', '--quiet', '--force', '--input', xclbin, '--dump-section', f'CLOCK_FREQ_TOPOLOGY:JSON:{path}']
        if subprocess.run(cmd, capture_output=True).returncode != 0 or not os.path.isfile(path):
            return None
        with open(path) as f:
            clocks = json.load(f).get('clock_freq_topology', {}).get('m_clock_freq', [])
    return next((int(c['m_freq_Mhz']) for c in clocks if c.get('m_name') == 'DATA_CLK'), None)


def _parse_kernel_utilization(util_file, kernel_name):
    """Read the resources of one kernel from a report written by Vivado's report_accelerator_utilization.

    The link step of v++ writes this report for the kernels of the linked design. Vivado counts block RAM in tiles of
    36 Kb; it is returned in blocks of 18 Kb, the unit of the C synthesis report.
    """

    columns = {'LUT': 'LUT', 'REG': 'FF', 'BRAM': 'BRAM_18K', 'URAM': 'URAM', 'DSP': 'DSP48E'}
    header = None
    with open(util_file) as f:
        for line in f:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if not cells:
                continue
            if header is None:
                if cells[0] == 'Name':
                    header = cells
            elif cells[0] == kernel_name:
                report = {}
                for name, cell in zip(header[1:], cells[1:]):
                    if name in columns:
                        # A cell holds the count and the share of the device, as in '6869 [  2.55%]'
                        value = cell.split()[0]
                        if name == 'BRAM':
                            value = f'{float(value) * 2:g}'
                        report[columns[name]] = value
                return report
    return {}


def _parse_bitfile_reports(report_dir, kernel_name):
    """Read the reports of the linked design that link_system.sh copies to final_reports.

    Returns the resources of the kernel, the slack of the whole design and the power of the whole device, or None if
    there is no report of the kernel's resources.
    """

    util_file = os.path.join(report_dir, 'impl_1_kernel_util_routed.rpt')
    if not os.path.isfile(util_file):
        return None
    report = _parse_kernel_utilization(util_file, kernel_name)
    # The name of the timing summary depends on the platform, so it is read only when there is exactly one
    timing_files = glob.glob(os.path.join(report_dir, 'impl_1_*_timing_summary_routed.rpt'))
    if len(timing_files) == 1:
        report.update(_parse_vivado_timing_summary(timing_files[0]))
    power_file = os.path.join(report_dir, 'impl_1_power_routed.rpt')
    if os.path.isfile(power_file):
        report.update(_parse_vivado_power(power_file))
    return report


def parse_vitis_unified_report(hls_dir):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    config_file = os.path.join(hls_dir, 'hls4ml_config.yml')
    if not os.path.isfile(config_file):
        print('Unable to read project data. Exiting.')
        return

    with open(config_file) as f:
        config = yaml.load(f, Loader=_ConfigLoader)
    project_name = config['ProjectName']
    top_func_name = f'{project_name}_{config["VitisUnifiedConfig"]["axi_mode"]}'
    hls_prj_dir = os.path.join(hls_dir, 'vitis_workspace', project_name, 'vitis_unified_project', 'hls')

    report = {}

    syn_file = os.path.join(hls_prj_dir, 'syn', 'report', f'{top_func_name}_csynth.xml')
    if os.path.isfile(syn_file):
        report['CSynthesisReport'] = _parse_csynth_xml(syn_file)
    else:
        print('CSynthesis report not found.')

    cosim_file = os.path.join(hls_prj_dir, 'sim', 'report', f'{top_func_name}_cosim.rpt')
    if os.path.isfile(cosim_file):
        report['CosimReport'] = _parse_cosim_rpt(cosim_file)
    else:
        print('Cosim report not found.')

    vivado_synth_report = _parse_vivado_synth_reports(hls_dir, 'vivado_synth')
    if vivado_synth_report is not None:
        report['VivadoSynthReport'] = vivado_synth_report

    vivado_impl_report = _parse_vivado_synth_reports(hls_dir, 'vivado_impl')
    if vivado_impl_report is not None:
        report['VivadoImplReport'] = vivado_impl_report

    bitfile_report = _parse_bitfile_reports(os.path.join(hls_dir, 'final_reports'), top_func_name)
    if bitfile_report is not None:
        report['BitfileReport'] = bitfile_report

    xclbin = os.path.join(hls_dir, 'vitis_workspace', 'system_link', f'{project_name}.xclbin')
    kernel_clock = read_kernel_clock(xclbin)
    if kernel_clock is not None:
        report['KernelClockMHz'] = kernel_clock

    return report
