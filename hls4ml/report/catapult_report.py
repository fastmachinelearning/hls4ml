import os
import re

import yaml


def read_catapult_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/build_prj.tcl'):
        prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')
        print('Prj Dir:', prj_dir)
        print('Top func name:', top_func_name)

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    solutions = _find_solutions(sln_dir, hls_dir)

    for sln in solutions:
        print(f'Reports for solution "{sln}":\n')
        _find_reports(sln_dir + '/' + sln, top_func_name, full_report)


def _parse_build_script(script_path):
    prj_dir = None
    top_func_name = None

    with open(script_path) as f:
        for line in f.readlines():
            if 'project new' in line:
                prj_dir = line.split()[-1]
            if 'set design_top' in line:
                top_func_name = line.split()[-1]

    return prj_dir, top_func_name


def _find_solutions(sln_dir, hls_dir):
    solutions = []
    prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')
    for path in os.listdir(sln_dir):
        # check if current path is a dir
        if os.path.isdir(os.path.join(sln_dir, path)):
            pathstring = str(path)
            if top_func_name in pathstring:
                solutions.append(pathstring)
    return solutions


def _find_reports(sln_dir, top_func_name, full_report=False):
    csim_file = sln_dir + '/../../tb_data/csim_results.log'
    if os.path.isfile(csim_file):
        _show_csim_report(csim_file)
    else:
        print('C simulation report not found.')

    syn_file = sln_dir + '/rtl.rpt'
    if os.path.isfile(syn_file):
        _show_synth_report(syn_file, full_report)
    else:
        print('Synthesis report not found.')

    cosim_file = sln_dir + f'/sim/report/{top_func_name}_cosim.rpt'
    if os.path.isfile(cosim_file):
        _show_cosim_report(cosim_file)
    else:
        print('Co-simulation report not found.')

    timing_report = sln_dir + '/vivado_concat_v/timing_summary_synth.rpt'
    if os.path.isfile(timing_report):
        _show_timing_report(timing_report)
    else:
        print('Timing synthesis report not found.')

    utilization_report = sln_dir + '/vivado_concat_v/utilization_synth.rpt'
    if os.path.isfile(utilization_report):
        _show_utilization_report(utilization_report)
    else:
        print('Utilization synthesis report not found.')


def _show_csim_report(csim_file):
    with open(csim_file) as f:
        print('C SIMULATION RESULT:')
        print(f.read())


def _show_synth_report(synth_file, full_report=False):
    with open(synth_file) as f:
        print('SYNTHESIS REPORT:')
        for line in f.readlines()[2:]:
            if not full_report and '* DSP48' in line:
                break
            print(line, end='')


def _show_cosim_report(cosim_file):
    with open(cosim_file) as f:
        print('CO-SIMULATION RESULT:')
        print(f.read())


def _show_timing_report(timing_report):
    with open(timing_report) as f:
        print('TIMING REPORT:')
        print(f.read())


def _show_utilization_report(utilization_report):
    with open(utilization_report) as f:
        print('UTILIZATION REPORT:')
        print(f.read())


def _get_abs_and_percentage_values(unparsed_cell):
    return int(unparsed_cell.split('(')[0]), float(unparsed_cell.split('(')[1].replace('%', '').replace(')', ''))


def parse_catapult_report(output_dir):
    if not os.path.exists(output_dir):
        print(f'Project OutputDir {output_dir} does not exist. Exiting.')
        return

    # Read the YAML config file to determine the project settings
    with open(output_dir + '/hls4ml_config.yml') as yfile:
        ydata = yaml.safe_load(yfile)

    if not ydata['ProjectDir'] is None:
        ProjectDir = ydata['ProjectDir']
    else:
        ProjectDir = ydata['ProjectName'] + '_prj'
    ProjectName = ydata['ProjectName']

    sln_dir = output_dir + '/' + ProjectDir
    if not os.path.exists(sln_dir):
        print(f'Project {ProjectDir} does not exist. Rerun "hls4ml build -p {output_dir}".')
        return

    solutions = _find_solutions(sln_dir, output_dir)
    if len(solutions) > 1:
        print(f'WARNING: Found {len(solutions)} solution(s) in {sln_dir}. Using the first solution.')

    report = {}

    sim_file = output_dir + '/tb_data/csim_results.log'
    if os.path.isfile(sim_file):
        csim_results = []
        with open(sim_file) as f:
            for line in f.readlines():
                csim_results.append([r for r in line.split()])
        report['CSimResults'] = csim_results

    util_report_file = output_dir + '/' + ProjectDir + '/' + solutions[0] + '/vivado_concat_v/utilization_synth.rpt'
    if os.path.isfile(util_report_file):
        util_report = {}
        a = 0
        with open(util_report_file) as f:
            for line in f.readlines():
                # Sometimes, phrases such as 'CLB Registers' can show up in the non-tabular sections of the report
                if '|' in line:
                    if ('CLB LUTs' in line) and (a == 0):
                        a += 1
                        util_report['LUT'] = line.split('|')[2].strip()
                    elif ('CLB Registers' in line) and (a == 1):
                        a += 1
                        util_report['FF'] = line.split('|')[2].strip()
                    elif ('RAMB18 ' in line) and (a == 2):
                        a += 1
                        util_report['BRAM_18K'] = line.split('|')[2].strip()
                    elif ('DSPs' in line) and (a == 3):
                        a += 1
                        util_report['DSP48E'] = line.split('|')[2].strip()
                    elif ('URAM' in line) and (a == 4):
                        a += 1
                        util_report['URAM'] = line.split('|')[2].strip()
        report['UtilizationReport'] = util_report
    else:
        print('Utilization report not found.')

    timing_report_file = output_dir + '/' + ProjectDir + '/' + solutions[0] + '/vivado_concat_v/timing_summary_synth.rpt'
    if os.path.isfile(timing_report_file):
        timing_report = {}
        with open(timing_report_file) as f:
            while not re.search('WNS', next(f)):
                pass
            # skip the successive line
            next(f)
            result = next(f).split()

        timing_report['WNS'] = float(result[0])
        timing_report['TNS'] = float(result[1])
        timing_report['WHS'] = float(result[4])
        timing_report['THS'] = float(result[5])
        timing_report['WPWS'] = float(result[8])
        timing_report['TPWS'] = float(result[9])

        report['TimingReport'] = timing_report
    else:
        print('Timing report not found.')

    latest_prj_dir = get_latest_project_prj_directory(output_dir, ProjectDir)
    latest_ver_dir = get_latest_project_version_directory(latest_prj_dir, ProjectName)
    file_path = os.path.join(latest_ver_dir, 'nnet_layer_results.txt')
    print('Results in nnet_layer_results.txt from:', file_path)

    # Initialize the array
    report['PerLayerQOFR'] = []
    # Open the file and read its contents
    with open(file_path) as file:
        # Read each line and append it to the list
        for line in file:
            report['PerLayerQOFR'].append(line.strip())  # strip() removes leading/trailing

    return report


def get_latest_project_version_directory(base_path, ProjectName):
    versions = [d for d in os.listdir(base_path) if d.startswith(ProjectName + '.v')]
    if not versions:
        raise FileNotFoundError('Error: No versions found.')
    latest_version = max(versions)
    return os.path.join(base_path, latest_version)


def get_latest_project_prj_directory(base_path, ProjectDir):
    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith(ProjectDir)]
    if not versions:
        raise FileNotFoundError('Error: No versions found.')
    latest_version = max(versions)
    return os.path.join(base_path, latest_version)


def qofr(report):
    # Access the PerLayerQOFR list from the report dictionary
    PerLayerQOFR = report.get('PerLayerQOFR', [])

    # Check if the list is not empty
    if PerLayerQOFR:
        # print('Results in nnet_layer_results.txt:')
        # Iterate over each line in the list and print it
        for line in PerLayerQOFR:
            print(line)
    else:
        print('No results found in nnet_layer_results.txt')
