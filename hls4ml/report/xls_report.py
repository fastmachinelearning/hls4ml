import os
import re
from pathlib import Path


def _parse_project(path) -> tuple[str, str]:
    prj_dir = None
    top_func_name = None

    project_path = Path(path + "/firmware")
    sv_files = list(project_path.glob("*.x"))
    project_file = sv_files[0]

    top_func_name = project_file.stem
    prj_dir = top_func_name + '_prj'

    return prj_dir, top_func_name


def parse_xls_report(hls_dir) -> dict:
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    prj_dir = None
    top_func_name = None

    prj_dir, top_func_name = _parse_project(hls_dir)

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    report = {}

    vivado_syn_file = hls_dir + '/reports/synth_util.rpt'
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
        print('Vivado synthesis report not found.')

    return report