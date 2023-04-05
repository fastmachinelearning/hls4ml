import os
import re
import sys
import xml.etree.ElementTree as ET


def read_vivado_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
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
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    solutions = _find_solutions(sln_dir)
    print(f'Found {len(solutions)} solution(s) in {sln_dir}.')

    for sln in solutions:
        print(f'Reports for solution "{sln}":\n')
        _find_reports(sln_dir + '/' + sln, top_func_name, full_report)


def _parse_project_script(path):
    prj_dir = None
    top_func_name = None

    project_path = path + '/project.tcl'

    with open(project_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'
            if 'set backend' in line:
                backend_name = line.split('"')[-2]

    if 'accelerator' in backend_name:
        top_func_name += '_axi'

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
    csim_file = sln_dir + f'/csim/report/{top_func_name}_csim.log'
    if os.path.isfile(csim_file):
        _show_csim_report(csim_file)
    else:
        print('C simulation report not found.')

    syn_file = sln_dir + f'/syn/report/{top_func_name}_csynth.rpt'
    if os.path.isfile(syn_file):
        _show_synth_report(syn_file, full_report)
    else:
        print('Synthesis report not found.')

    cosim_file = sln_dir + f'/sim/report/{top_func_name}_cosim.rpt'
    if os.path.isfile(cosim_file):
        _show_cosim_report(cosim_file)
    else:
        print('Co-simulation report not found.')


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


def _get_abs_and_percentage_values(unparsed_cell):
    return int(unparsed_cell.split('(')[0]), float(unparsed_cell.split('(')[1].replace('%', '').replace(')', ''))


def parse_vivado_report(hls_dir):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
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
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    solutions = _find_solutions(sln_dir)
    if len(solutions) > 1:
        print(f'WARNING: Found {len(solutions)} solution(s) in {sln_dir}. Using the first solution.')

    report = {}

    sim_file = hls_dir + '/tb_data/csim_results.log'
    if os.path.isfile(sim_file):
        csim_results = []
        with open(sim_file) as f:
            for line in f.readlines():
                csim_results.append([r for r in line.split()])
        report['CSimResults'] = csim_results

    sim_file = hls_dir + '/tb_data/rtl_cosim_results.log'
    if os.path.isfile(sim_file):
        cosim_results = []
        with open(sim_file) as f:
            for line in f.readlines():
                cosim_results.append([r for r in line.split()])
        report['CosimResults'] = cosim_results

    syn_file = sln_dir + '/' + solutions[0] + f'/syn/report/{top_func_name}_csynth.xml'
    c_synth_report = {}
    if os.path.isfile(syn_file):
        root = ET.parse(syn_file).getroot()

        # Performance
        perf_node = root.find('./PerformanceEstimates')
        c_synth_report['TargetClockPeriod'] = root.find('./UserAssignments/TargetClockPeriod').text
        c_synth_report['EstimatedClockPeriod'] = perf_node.find('./SummaryOfTimingAnalysis/EstimatedClockPeriod').text
        c_synth_report['BestLatency'] = perf_node.find('./SummaryOfOverallLatency/Best-caseLatency').text
        c_synth_report['WorstLatency'] = perf_node.find('./SummaryOfOverallLatency/Worst-caseLatency').text
        c_synth_report['IntervalMin'] = perf_node.find('./SummaryOfOverallLatency/Interval-min').text
        c_synth_report['IntervalMax'] = perf_node.find('./SummaryOfOverallLatency/Interval-max').text
        # Area
        area_node = root.find('./AreaEstimates')
        for child in area_node.find('./Resources'):
            # DSPs are called 'DSP48E' in Vivado and just 'DSP' in Vitis. Overriding here to have consistent keys
            if child.tag == 'DSP48E':
                child.tag = 'DSP'
            c_synth_report[child.tag] = child.text
        for child in area_node.find('./AvailableResources'):
            if child.tag == 'DSP48E':
                child.tag = 'DSP'
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

    cosim_file = sln_dir + '/' + solutions[0] + f'/sim/report/{top_func_name}_cosim.rpt'
    if os.path.isfile(cosim_file):
        cosim_report = {}
        with open(cosim_file) as f:
            for line in f.readlines():
                if re.search('VHDL', line) or re.search('Verilog', line):
                    result = line[1:].split()  # [1:] skips the leading '|'
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
        transaction_file = (
            sln_dir
            + '/'
            + solutions[0]
            + '/sim/'
            + report['CosimReport']['RTL'].lower()
            + '/'
            + top_func_name
            + '.performance.result.transaction.xml'
        )
        if os.path.isfile(transaction_file):
            cosim_transactions = {
                'InitiationInterval': {'max': 0, 'min': sys.maxsize, 'avg': 0.0},
                'Latency': {'max': 0, 'min': sys.maxsize, 'avg': 0.0},
            }
            with open(transaction_file) as f:
                i = 1
                for line in f.readlines():
                    if re.search('transaction', line):
                        result = line.split()
                        # update min
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['min'] = (
                                int(result[3])
                                if int(result[3]) < cosim_transactions['InitiationInterval']['min']
                                else cosim_transactions['InitiationInterval']['min']
                            )
                        cosim_transactions['Latency']['min'] = (
                            int(result[2])
                            if int(result[2]) < cosim_transactions['Latency']['min']
                            else cosim_transactions['Latency']['min']
                        )
                        # update max
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['max'] = (
                                int(result[3])
                                if int(result[3]) > cosim_transactions['InitiationInterval']['max']
                                else cosim_transactions['InitiationInterval']['max']
                            )
                        cosim_transactions['Latency']['max'] = (
                            int(result[2])
                            if int(result[2]) > cosim_transactions['Latency']['max']
                            else cosim_transactions['Latency']['max']
                        )
                        # update avg
                        if result[3] != 'x':
                            cosim_transactions['InitiationInterval']['avg'] = cosim_transactions['InitiationInterval'][
                                'avg'
                            ] + float((int(result[3]) - cosim_transactions['InitiationInterval']['avg']) / i)
                        cosim_transactions['Latency']['avg'] = cosim_transactions['Latency']['avg'] + float(
                            (int(result[2]) - cosim_transactions['Latency']['avg']) / i
                        )
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
            with open(util_rpt_file) as f:
                for line in f.readlines():
                    if re.search(r'\(top\)', line):
                        # Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs | RAMB36 | RAMB18 (|   URAM   )| DSP48 Blocks
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

    timing_report_file = (
        hls_dir
        + '/'
        + prj_dir.split('_')[0]
        + '_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper_timing_summary_routed.rpt'
    )
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
    return report


def print_vivado_report(report_dict):
    if _is_running_in_notebook():
        _print_ipython_report(report_dict)
    else:
        _print_str_report(report_dict)


def _print_ipython_report(report_dict):
    from IPython.display import HTML, display

    html = '<html>\n' + _table_css + '<div class="hls4ml">'
    body = _make_report_body(report_dict, _make_html_table_template, _make_html_header)
    html += body + '\n</div>\n</html>'
    display(HTML(html))


def _print_str_report(report_dict):
    body = _make_report_body(report_dict, _make_str_table_template, _make_str_header)
    print(body)


def _is_running_in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


_table_css = """
<style>
.hls4ml {
    font-family: Tahoma, Geneva, sans-serif;
}
.hls4ml h3 {
    font-size: 15px;
    font-weight: 600;
    color: #54585d;
}
.hls4ml table {
    border-collapse: collapse;
    display: inline-block;
    padding: 2px;
}
.hls4ml table td {
    text-align: left;
    padding: 10px;
}
.hls4ml table td:nth-child(2) {
    text-align: right;
}
.hls4ml table thead td {
    background-color: #54585d;
    color: #ffffff;
    font-weight: bold;
    font-size: 11px;
    border: 1px solid #54585d;
}
.hls4ml table tbody td {
    color: #636363;
    border: 1px solid #dddfe1;
    font-size: 11px;
}
.hls4ml table tbody tr {
    background-color: #f9fafb;
}
.hls4ml table tbody tr:nth-child(odd) {
    background-color: #ffffff;
}
</style>
"""

_table_base_template = """
<table>
    <thead>
        <tr>
            <td colspan=2>{table_header}</td>
        </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
</table>
"""

_row_base_template = "        <tr><td>{row_title}</td><td>{{{row_key}}}</td>"


def _make_html_table_template(table_header, row_templates):
    table_rows = '\n'.join(
        [_row_base_template.format(row_title=row_title, row_key=row_key) for row_title, row_key in row_templates.items()]
    )
    return _table_base_template.format(table_header=table_header, table_rows=table_rows)


def _make_str_table_template(table_header, row_templates):
    len_title = 0
    for row_title in row_templates.keys():
        if len(row_title) > len_title:
            len_title = len(row_title)
    head = f'\n - {table_header}:\n'
    table_rows = '\n'.join(
        ['    ' + f'{row_title}:'.ljust(len_title + 1) + f' {{{row_key}}}' for row_title, row_key in row_templates.items()]
    )
    return head + table_rows + '\n'


def _make_html_header(report_header):
    return f'<h3>{report_header}:</h3>'


def _make_str_header(report_header):
    sep = '=' * 54 + '\n'
    return '\n' + sep + '== ' + report_header + '\n' + sep


def _convert_cycles_to_time(n_cycles, clock_period):
    time_in_ns = n_cycles * clock_period
    if time_in_ns < 1000:
        return str(time_in_ns) + ' ns'

    time_in_us = time_in_ns / 1000
    if time_in_us < 1000:
        return str(time_in_us) + ' \u00B5s'

    time_in_ms = time_in_us / 1000
    if time_in_ms < 1000:
        return str(time_in_ms) + ' ms'

    time_in_s = time_in_ms / 1000
    if time_in_s < 1000:
        return str(time_in_s) + ' s'


def _make_report_body(report_dict, make_table_template, make_header_template):
    body = ''

    if 'CSynthesisReport' in report_dict:
        body += make_header_template('C Synthesis report')
        perf_rows = {
            'Best-case latency': 'best_latency',
            'Worst-case latency': 'worst_latency',
            'Interval Min': 'interval_min',
            'Interval Max': 'interval_max',
            'Estimated Clock Period': 'estimated_clock',
        }
        area_rows = {
            'BRAM_18K': 'bram',
            'DSP': 'dsp',
            'FF': 'ff',
            'LUT': 'lut',
            'URAM': 'uram',
        }
        body += make_table_template('Performance estimates', perf_rows)
        body += make_table_template('Resource estimates', area_rows)

        csynth_report = report_dict['CSynthesisReport']
        target_clock = float(csynth_report['TargetClockPeriod'])
        best_latency = int(csynth_report['BestLatency'])
        worst_latency = int(csynth_report['BestLatency'])
        bram = int(csynth_report['BRAM_18K'])
        avail_bram = int(csynth_report['AvailableBRAM_18K'])
        dsp = int(csynth_report['DSP'])
        avail_dsp = int(csynth_report['AvailableDSP'])
        ff = int(csynth_report['FF'])
        avail_ff = int(csynth_report['AvailableFF'])
        lut = int(csynth_report['LUT'])
        avail_lut = int(csynth_report['AvailableLUT'])
        if 'URAM' in csynth_report:
            uram = int(csynth_report['URAM'])
            avail_uram = int(csynth_report['AvailableURAM'])

        params = {}

        params['best_latency'] = str(best_latency) + ' (' + _convert_cycles_to_time(best_latency, target_clock) + ')'
        params['worst_latency'] = str(worst_latency) + ' (' + _convert_cycles_to_time(worst_latency, target_clock) + ')'
        params['interval_min'] = csynth_report['IntervalMin']
        params['interval_max'] = csynth_report['IntervalMax']
        params['estimated_clock'] = csynth_report['EstimatedClockPeriod']

        params['bram'] = str(bram) + ' / ' + str(avail_bram) + ' (' + str(round(bram / avail_bram * 100, 1)) + '%)'
        params['dsp'] = str(dsp) + ' / ' + str(avail_dsp) + ' (' + str(round(dsp / avail_dsp * 100, 1)) + '%)'
        params['ff'] = str(ff) + ' / ' + str(avail_ff) + ' (' + str(round(ff / avail_ff * 100, 1)) + '%)'
        params['lut'] = str(lut) + ' / ' + str(avail_lut) + ' (' + str(round(lut / avail_lut * 100, 1)) + '%)'
        if 'URAM' in csynth_report and avail_uram > 0:
            params['uram'] = str(uram) + ' / ' + str(avail_uram) + ' (' + str(round(uram / avail_uram * 100, 1)) + '%)'
        else:
            params['uram'] = 'N/A'

        body = body.format(**params)

    if 'VivadoSynthReport' in report_dict:
        body += make_header_template('Vivado Synthesis report')
        area_rows = {
            'BRAM_18K': 'bram',
            'DSP48E': 'dsp',
            'FF': 'ff',
            'LUT': 'lut',
            'URAM': 'uram',
        }
        body += make_table_template('Resource utilization', area_rows)

        vsynth_report = report_dict['VivadoSynthReport']

        params = {}
        params['bram'] = vsynth_report['BRAM_18K']
        params['dsp'] = vsynth_report['DSP48E']
        params['ff'] = vsynth_report['FF']
        params['lut'] = vsynth_report['LUT']
        params['uram'] = vsynth_report['URAM'] if 'URAM' in vsynth_report else 'N/A'

        body = body.format(**params)

    if 'CosimReport' in report_dict:
        body += make_header_template('Co-Simulation report')
        perf_rows = {
            'Status': 'status',
            'Best-case latency': 'best_latency',
            'Worst-case latency': 'worst_latency',
            'Interval Min': 'interval_min',
            'Interval Max': 'interval_max',
        }
        body += make_table_template('Performance', perf_rows)

        cosim_report = report_dict['CosimReport']

        params = {}
        params['status'] = cosim_report['Status']
        params['best_latency'] = cosim_report['LatencyMin']
        params['worst_latency'] = cosim_report['LatencyMax']
        params['interval_min'] = cosim_report['IntervalMin']
        params['interval_max'] = cosim_report['IntervalMax']

        body = body.format(**params)

    if 'ImplementationReport' in report_dict:
        body += make_header_template('Implementation report')
        area_rows = {
            'Total LUTs': 'lut',
            'Logic LUTs': 'logiclut',
            'LUTRAM': 'lutram',
            'SRLs': 'srl',
            'FF': 'ff',
            'RAMB18': 'bram18',
            'RAMB36': 'bram36',
            'DSP': 'dsp',
            'URAM': 'uram',
        }
        body += make_table_template('Resource utilization', area_rows)

        impl_report = report_dict['ImplementationReport']

        params = {}
        params['lut'] = impl_report['TotLUTs'] + ' (' + impl_report['TotLUTs%'] + '%)'
        params['logiclut'] = impl_report['LogicLUTs'] + ' (' + impl_report['LogicLUTs%'] + '%)'
        params['lutram'] = impl_report['LUTRAMs'] + ' (' + impl_report['LUTRAMs%'] + '%)'
        params['srl'] = impl_report['SRLs'] + ' (' + impl_report['SRLs%'] + '%)'
        params['ff'] = impl_report['FFs'] + ' (' + impl_report['FFs%'] + '%)'
        params['bram18'] = impl_report['RAMB18s'] + ' (' + impl_report['RAMB18s%'] + '%)'
        params['bram36'] = impl_report['RAMB36s'] + ' (' + impl_report['RAMB36s%'] + '%)'
        params['dsp'] = impl_report['DSPs'] + ' (' + impl_report['DSPs%'] + '%)'
        if 'URAMs' in impl_report:
            params['uram'] = impl_report['URAMs'] + ' (' + impl_report['URAMs%'] + '%)'
        else:
            params['uram'] = 'N/A'

        body = body.format(**params)

    if 'TimingReport' in report_dict:
        body += make_header_template('Timing report')
        perf_rows = {
            'Worst Negative Slack (WNS)': 'wns',
            'Total Negative Slack (TNS)': 'tns',
            'Worst Hold Slack (WHS)': 'whs',
            'Total Hold Slack (THS)': 'ths',
            'Worst Pulse Width Slack (WPWS)': 'wpws',
            'Total Pulse Width Slack (TPWS)': 'tpws',
        }
        body += make_table_template('Timing', perf_rows)

        timing_report = report_dict['TimingReport']

        params = {}
        params['wns'] = round(timing_report['WNS'], 2)
        params['tns'] = round(timing_report['TNS'], 2)
        params['whs'] = round(timing_report['WHS'], 2)
        params['ths'] = round(timing_report['THS'], 2)
        params['wpws'] = round(timing_report['WPWS'], 2)
        params['tpws'] = round(timing_report['TPWS'], 2)

        body = body.format(**params)

    return body
