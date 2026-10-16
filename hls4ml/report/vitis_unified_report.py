import json
import os
import shutil
import subprocess
import tempfile

import yaml

from hls4ml.report.vivado_report import _parse_cosim_rpt, _parse_csynth_xml


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

    xclbin = os.path.join(hls_dir, 'vitis_workspace', 'system_link', f'{project_name}.xclbin')
    kernel_clock = read_kernel_clock(xclbin)
    if kernel_clock is not None:
        report['KernelClockMHz'] = kernel_clock

    return report
