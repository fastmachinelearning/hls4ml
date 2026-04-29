import os
import re
from pathlib import Path


def _get_project_name(path) -> str:
    project_path = Path(path + "/firmware")
    sv_files = list(project_path.glob("*.sv"))
    return sv_files[0].stem


def parse_xls_report(hls_dir) -> dict:
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return {}

    project_name = _get_project_name(hls_dir)
    report_dir = Path(hls_dir) / f'output_{project_name}' / 'reports'

    vivado_syn_file = report_dir / f'{project_name}_post_synth_util.rpt'
    report = {}
    if os.path.isfile(vivado_syn_file):
        vivado_synth_rpt = {}
        with open(vivado_syn_file) as f:
            section = 0
            for line in f.readlines():
                match = re.match(r'^(\d)\.', line)
                if match:
                    section = int(match.group(1))
                # Sometimes, phrases such as 'CLB Registers' can show up in the non-tabular sections of the report
                if '|' in line:
                    # CLB (2019.X) vs. Slice (2020.X)
                    if ('CLB LUTs' in line or 'Slice LUTs' in line) and section == 1:
                        vivado_synth_rpt['LUT'] = line.split('|')[2].strip()
                    elif ('CLB Registers' in line or 'Slice Registers' in line) and section == 1:
                        vivado_synth_rpt['FF'] = line.split('|')[2].strip()
                    elif 'Block RAM Tile' in line and section == 2:
                        vivado_synth_rpt['BRAM_18K'] = line.split('|')[2].strip()
                    elif 'URAM' in line and section == 2:
                        vivado_synth_rpt['URAM'] = line.split('|')[2].strip()
                    elif 'DSPs' in line and section == 3:
                        vivado_synth_rpt['DSP48E'] = line.split('|')[2].strip()
        report['VivadoSynthReport'] = vivado_synth_rpt
    else:
        print(f'Vivado synthesis report not found at {vivado_syn_file}.')

    return report
