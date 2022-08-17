from __future__ import print_function
import os
import re
import sys
import xml.etree.ElementTree as ET

def read_vivado_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/project.tcl'):
        prj_dir, top_func_name = _parse_project_script(hls_dir)

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return

    solutions = _find_solutions(sln_dir)
    print('Found {} solution(s) in {}.'.format(len(solutions), sln_dir))

    for sln in solutions:
        print('Reports for solution "{}":\n'.format(sln))
        _find_reports(sln_dir + '/' + sln, top_func_name, full_report)

def _parse_project_script(path):
    prj_dir = None
    top_func_name = None

    project_path = path + '/project.tcl'

    with open(project_path, 'r') as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'

    return prj_dir, top_func_name

def _find_solutions(sln_dir):
    solutions = []

    if os.path.isfile(sln_dir + '/vivado_hls.app'):
        sln_file = 'vivado_hls.app'
    elif os.path.isfile(sln_dir + '/hls.app'):
        sln_file = 'hls.app'
    else:
        return solutions

    with open(sln_dir + '/' + sln_file) as f:
        # Get rid of namespaces (workaround to support two types of vivado_hls.app files)
        xmlstring = re.sub(' xmlns="[^"]+"', '', f.read(), count=1)

    root = ET.fromstring(xmlstring)
    for sln_tag in root.findall('solutions/solution'):
        sln_name = sln_tag.get('name')
        if sln_name is not None and os.path.isdir(sln_dir + '/' + sln_name):
            solutions.append(sln_name)

    return solutions

def _find_reports(sln_dir, top_func_name, full_report=False):
    csim_file = sln_dir + '/csim/report/{}_csim.log'.format(top_func_name)
    if os.path.isfile(csim_file):
        _show_csim_report(csim_file)
    else:
        print('C simulation report not found.')

    syn_file = sln_dir + '/syn/report/{}_csynth.rpt'.format(top_func_name)
    if os.path.isfile(syn_file):
        _show_synth_report(syn_file, full_report)
    else:
        print('Synthesis report not found.')

    cosim_file = sln_dir + '/sim/report/{}_cosim.rpt'.format(top_func_name)
    if os.path.isfile(cosim_file):
        _show_cosim_report(cosim_file)
    else:
        print('Co-simulation report not found.')

def _show_csim_report(csim_file):
    with open(csim_file, 'r') as f:
        print('C SIMULATION RESULT:')
        print(f.read())

def _show_synth_report(synth_file, full_report=False):
    with open(synth_file, 'r') as f:
        print('SYNTHESIS REPORT:')
        for line in f.readlines()[2:]:
            if not full_report and '* DSP48' in line:
                break
            print(line, end = '')

def _show_cosim_report(cosim_file):
    with open(cosim_file, 'r') as f:
        print('CO-SIMULATION RESULT:')
        print(f.read())

def _get_abs_and_percentage_values(unparsed_cell):
    return int(unparsed_cell.split('(')[0]), float(unparsed_cell.split('(')[1].replace('%', '').replace(')', ''))

def parse_vivado_report(hls_dir):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/project.tcl'):
        prj_dir, top_func_name = _parse_project_script(hls_dir)

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = hls_dir + '/' + prj_dir
    if not os.path.exists(sln_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return

    solutions = _find_solutions(sln_dir)
    if len(solutions) > 1:
        print('WARNING: Found {} solution(s) in {}. Using the first solution.'.format(len(solutions), sln_dir))

    report = {}

    sim_file = hls_dir + '/tb_data/csim_results.log'
    if os.path.isfile(sim_file):
        csim_results = []
        with open(sim_file, 'r') as f:
            for line in f.readlines():
                csim_results.append([r for r in line.split()])
        report['CSimResults'] = csim_results

    sim_file = hls_dir + '/tb_data/rtl_cosim_results.log'
    if os.path.isfile(sim_file):
        cosim_results = []
        with open(sim_file, 'r') as f:
            for line in f.readlines():
                cosim_results.append([r for r in line.split()])
        report['CosimResults'] = cosim_results

    syn_file = sln_dir + '/' + solutions[0] + '/syn/report/{}_csynth.xml'.format(top_func_name)
    c_synth_report = {}
    if os.path.isfile(syn_file):
        root = ET.parse(syn_file).getroot()

        # Performance
        perf_node = root.find('./PerformanceEstimates')
        c_synth_report['EstimatedClockPeriod'] = perf_node.find('./SummaryOfTimingAnalysis/EstimatedClockPeriod').text
        c_synth_report['BestLatency'] = perf_node.find('./SummaryOfOverallLatency/Best-caseLatency').text
        c_synth_report['WorstLatency'] = perf_node.find('./SummaryOfOverallLatency/Worst-caseLatency').text
        c_synth_report['IntervalMin'] = perf_node.find('./SummaryOfOverallLatency/Interval-min').text
        c_synth_report['IntervalMax'] = perf_node.find('./SummaryOfOverallLatency/Interval-max').text
        # Area
        area_node = root.find('./AreaEstimates')
        for child in area_node.find('./Resources'):
            c_synth_report[child.tag] = child.text
        for child in area_node.find('./AvailableResources'):
            c_synth_report['Available' + child.tag] = child.text
        report['CSynthesisReport'] = c_synth_report
    else:
        print('CSynthesis report not found.')

    vivado_syn_file = hls_dir + '/vivado_synth.rpt'
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
                    if 'CLB LUTs' in line and section == 1:
                        vivado_synth_rpt['LUT'] = line.split('|')[2].strip()
                    elif 'CLB Registers' in line and section == 1:
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

    cosim_file = sln_dir + '/' + solutions[0] + '/sim/report/{}_cosim.rpt'.format(top_func_name)
    if os.path.isfile(cosim_file):
        cosim_report = {}
        with open(cosim_file, 'r') as f:
            for line in f.readlines():
                if re.search('VHDL', line) or re.search('Verilog', line):
                    result = line[1:].split() # [1:] skips the leading '|'
                    result = [res[:-1] if res[-1] == '|' else res for res in result]
                    # RTL, Status, Latency-min, Latency-avg, Latency-max, Interval-min, Interval-avg, Interval-max
                    if result[1] == 'NA':
                        continue
                    else:
                        cosim_report['RTL'] = result[0]
                        cosim_report['Status'] = result[1]
                        cosim_report['LatencyMin'] = result[2]
                        cosim_report['LatencyMax'] = result[4]
                        cosim_report['IntervalMin'] = result[5]
                        cosim_report['IntervalMax'] = result[7]
        report['CosimReport'] = cosim_report
    else:
        print('Cosim report not found.')

    if os.path.isfile(cosim_file):
        transaction_file = sln_dir + '/' + solutions[0] + '/sim/' + report['CosimReport']['RTL'].lower() + '/' + top_func_name + '.performance.result.transaction.xml'
        if os.path.isfile(transaction_file):
            cosim_transactions = {'InitiationInterval': {'max': 0, 'min': sys.maxsize, 'avg': 0.0},
                                  'Latency': {'max': 0, 'min': sys.maxsize, 'avg': 0.0}}
            with open(transaction_file, 'r') as f:
                i = 1
                for line in f.readlines():
                    if re.search('transaction', line):
                        result = line.split()
                        # update min
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['min'] = int(result[3]) if int(result[3]) < cosim_transactions['InitiationInterval']['min'] else cosim_transactions['InitiationInterval']['min']
                        cosim_transactions['Latency']['min'] = int(result[2]) if int(result[2]) < cosim_transactions['Latency']['min'] else cosim_transactions['Latency']['min']
                        # update max
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['max'] = int(result[3]) if int(result[3]) > cosim_transactions['InitiationInterval']['max'] else cosim_transactions['InitiationInterval']['max']
                        cosim_transactions['Latency']['max'] = int(result[2]) if int(result[2]) > cosim_transactions['Latency']['max'] else cosim_transactions['Latency']['max']
                        # update avg
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['avg'] = cosim_transactions['InitiationInterval']['avg'] + float((int(result[3]) - cosim_transactions['InitiationInterval']['avg']) / i)
                        cosim_transactions['Latency']['avg'] = cosim_transactions['Latency']['avg'] + float((int(result[2]) - cosim_transactions['Latency']['avg']) / i)
                        i += 1

            report['CosimReport']['LatencyMin'] = cosim_transactions['Latency']['min']
            report['CosimReport']['LatencyMax'] = cosim_transactions['Latency']['max']
            report['CosimReport']['LatencyAvg'] = cosim_transactions['Latency']['avg']

            report['CosimReport']['IntervalMin'] = cosim_transactions['InitiationInterval']['min']
            report['CosimReport']['IntervalMax'] = cosim_transactions['InitiationInterval']['max']
            report['CosimReport']['IntervalAvg'] = cosim_transactions['InitiationInterval']['avg']

        util_rpt_file = hls_dir + '/util.rpt'
        if os.path.isfile(util_rpt_file):
            implementation_report = {}
            with open(util_rpt_file, 'r') as f:
                for line in f.readlines():
                    if re.search('\(top\)', line):
                        # Total LUTs  |   Logic LUTs  |   LUTRAMs  |     SRLs    |      FFs      |    RAMB36   |   RAMB18  (|   URAM   )| DSP48 Blocks
                        # skipping the first 2 unuseful cells with [:2]
                        results = [_get_abs_and_percentage_values(elem) for elem in line.replace('|', '').split()[2:]]
                        implementation_report['TotLUTs'] = results[0][0]
                        implementation_report['TotLUTs%'] = results[0][1]

                        implementation_report['LogicLUTs'] = results[1][0]
                        implementation_report['LogicLUTs%'] = results[1][1]

                        implementation_report['LUTRAMs'] = results[2][0]
                        implementation_report['LUTRAMs%'] = results[2][1]

                        implementation_report['SRLs'] = results[3][0]
                        implementation_report['SRLs%'] = results[3][1]

                        implementation_report['FFs'] = results[4][0]
                        implementation_report['FFs%'] = results[4][1]

                        implementation_report['RAMB36s'] = results[5][0]
                        implementation_report['RAMB36s%'] = results[5][1]

                        implementation_report['RAMB18s'] = results[6][0]
                        implementation_report['RAMB18s%'] = results[6][1]

                        if len(results) == 9:
                            implementation_report['URAMs'] = results[7][0]
                            implementation_report['URAMs%'] = results[7][1]

                            implementation_report['DSPs'] = results[8][0]
                            implementation_report['DSPs%'] = results[8][1]
                        else:
                            implementation_report['DSPs'] = results[7][0]
                            implementation_report['DSPs%'] = results[7][1]
            report['ImplementationReport'] = implementation_report
        else:
            print('Implementation report not found.')

    timing_report_file = hls_dir + '/' + prj_dir.split('_')[0] + '_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper_timing_summary_routed.rpt'
    if os.path.isfile(timing_report_file):
        timing_report = {}
        with open(timing_report_file, 'r') as f:
            while not re.search('WNS', next(f)):
                pass
            # skip the successive line
            next(f)
            result = next(f).split()

        timing_report['WNS']  = float(result[0])
        timing_report['TNS']  = float(result[1])
        timing_report['WHS']  = float(result[4])
        timing_report['THS']  = float(result[5])
        timing_report['WPWS'] = float(result[8])
        timing_report['TPWS'] = float(result[9])

        report['TimingReport'] = timing_report
    else:
        print('Timing report not found.')
    return report

