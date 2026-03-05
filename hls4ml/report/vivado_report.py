import os
import re
import sys
import xml.etree.ElementTree as ET

# Path templates for report files. Use _path() to resolve with os.path.join.
# Placeholders: {hls_dir}, {prj_dir}, {sln_dir}, {solution}, {top}, {rtl}, {base}
PATHS = {
    'project_tcl': ('{hls_dir}', 'project.tcl'),
    'sln_dir': ('{hls_dir}', '{prj_dir}'),
    'vivado_hls_app': ('{sln_dir}', 'vivado_hls.app'),
    'hls_app': ('{sln_dir}', 'hls.app'),
    'solution_dir': ('{sln_dir}', '{solution}'),
    'csim_log': ('{sln_dir}', 'csim', 'report', '{top}_csim.log'),
    'csynth_rpt': ('{sln_dir}', 'syn', 'report', '{top}_csynth.rpt'),
    'cosim_rpt': ('{sln_dir}', 'sim', 'report', '{top}_cosim.rpt'),
    'csim_results': ('{hls_dir}', 'tb_data', 'csim_results.log'),
    'rtl_cosim_results': ('{hls_dir}', 'tb_data', 'rtl_cosim_results.log'),
    'csynth_xml': ('{sln_dir}', '{solution}', 'syn', 'report', '{top}_csynth.xml'),
    'vivado_synth': ('{hls_dir}', 'vivado_synth.rpt'),
    'cosim_report': ('{sln_dir}', '{solution}', 'sim', 'report', '{top}_cosim.rpt'),
    'transaction_file': (
        '{sln_dir}',
        '{solution}',
        'sim',
        '{rtl}',
        '{top}.performance.result.transaction.xml',
    ),
    'util_rpt_vivado': ('{hls_dir}', 'vivado_reports', 'post_route_util_hier.rpt'),
    'util_rpt_system': ('{hls_dir}', 'vivado_reports', 'post_route_util_hier_system.rpt'),
    'timing_summary_vivado': ('{hls_dir}', 'vivado_reports', 'post_route_timing_summary.rpt'),
    'timing_summary_system': ('{hls_dir}', 'vivado_reports', 'post_route_timing_summary_system.rpt'),
    'power_rpt_vivado': ('{hls_dir}', 'vivado_reports', 'post_route_power.rpt'),
    'power_rpt_system': ('{hls_dir}', 'vivado_reports', 'post_route_power_system.rpt'),
}

# Synthesis report (csynth.rpt)
SYNTH_HEADER_LINES = 2
SYNTH_TRUNCATE_MARKER = '* DSP48'

# Vivado synth report sections (numbered headers "1.", "2.", "3.")
VIVADO_SECTION_CLB = 1
VIVADO_SECTION_RAM = 2
VIVADO_SECTION_DSP = 3
VIVADO_COLUMN_INDEX = 2  # Resource value column after split by '|'

# Cosim .rpt table column indices (RTL, Status, Latency-min/avg/max, Interval-min/avg/max)
COSIM_COL_RTL = 0
COSIM_COL_STATUS = 1
COSIM_COL_LATENCY_MIN = 2
COSIM_COL_LATENCY_AVG = 3
COSIM_COL_LATENCY_MAX = 4
COSIM_COL_INTERVAL_MIN = 5
COSIM_COL_INTERVAL_AVG = 6
COSIM_COL_INTERVAL_MAX = 7

# Transaction file (performance.result.transaction.xml): latency and interval column indices
TX_LATENCY_IDX = 2
TX_INTERVAL_IDX = 3

# util report (top) line: cells to skip and column indices
UTIL_SKIP_CELLS = 2
UTIL_COL_TOTLUTS = 0
UTIL_COL_LOGICLUTS = 1
UTIL_COL_LUTRAMS = 2
UTIL_COL_SRLS = 3
UTIL_COL_FFS = 4
UTIL_COL_RAMB36 = 5
UTIL_COL_RAMB18 = 6
UTIL_COL_URAM = 7
UTIL_COL_DSP = 8
UTIL_COLS_WITH_URAM = 9
UTIL_COLS_WITHOUT_URAM = 8

# Timing summary report column indices
TIMING_COL_WNS = 0
TIMING_COL_TNS = 1
TIMING_COL_WHS = 4
TIMING_COL_THS = 5
TIMING_COL_WPWS = 8
TIMING_COL_TPWS = 9


def _path(name, **kwargs):
    """Build a path from PATHS template, resolving {placeholder} with kwargs."""
    segments = PATHS[name]
    resolved = []
    for seg in segments:
        for key, val in kwargs.items():
            seg = seg.replace('{' + key + '}', str(val))
        resolved.append(seg)
    return os.path.join(*resolved)


def read_vivado_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(_path('project_tcl', hls_dir=hls_dir)):
        prj_dir, top_func_name, _ = _parse_project_script(hls_dir)

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = _path('sln_dir', hls_dir=hls_dir, prj_dir=prj_dir)
    if not os.path.exists(sln_dir):
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    solutions = _find_solutions(sln_dir)
    print(f'Found {len(solutions)} solution(s) in {sln_dir}.')

    for sln in solutions:
        print(f'Reports for solution "{sln}":\n')
        _find_reports(_path('solution_dir', sln_dir=sln_dir, solution=sln), top_func_name, full_report)


def _parse_project_script(path):
    prj_dir = None
    top_func_name = None
    backend_name = 'vivado'

    project_path = _path('project_tcl', hls_dir=path)

    with open(project_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'
            if 'set backend' in line:
                backend_name = line.split('"')[-2]

    if 'accelerator' in backend_name:
        top_func_name += '_axi'

    return prj_dir, top_func_name, backend_name


def _find_solutions(sln_dir):
    solutions = []

    if os.path.isfile(_path('vivado_hls_app', sln_dir=sln_dir)):
        sln_file = 'vivado_hls.app'
    elif os.path.isfile(_path('hls_app', sln_dir=sln_dir)):
        sln_file = 'hls.app'
    else:
        return solutions

    with open(_path('vivado_hls_app' if sln_file == 'vivado_hls.app' else 'hls_app', sln_dir=sln_dir)) as f:
        # Get rid of namespaces (workaround to support two types of vivado_hls.app files)
        xmlstring = re.sub(' xmlns="[^"]+"', '', f.read(), count=1)

    root = ET.fromstring(xmlstring)
    for sln_tag in root.findall('solutions/solution'):
        sln_name = sln_tag.get('name')
        if sln_name is not None and os.path.isdir(_path('solution_dir', sln_dir=sln_dir, solution=sln_name)):
            solutions.append(sln_name)

    return solutions


def _find_reports(sln_dir, top_func_name, full_report=False):
    csim_file = _path('csim_log', sln_dir=sln_dir, top=top_func_name)
    if os.path.isfile(csim_file):
        _show_csim_report(csim_file)
    else:
        print('C simulation report not found.')

    syn_file = _path('csynth_rpt', sln_dir=sln_dir, top=top_func_name)
    if os.path.isfile(syn_file):
        _show_synth_report(syn_file, full_report)
    else:
        print('Synthesis report not found.')

    cosim_file = _path('cosim_rpt', sln_dir=sln_dir, top=top_func_name)
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
        for line in f.readlines()[SYNTH_HEADER_LINES:]:
            if not full_report and SYNTH_TRUNCATE_MARKER in line:
                break
            print(line, end='')


def _show_cosim_report(cosim_file):
    with open(cosim_file) as f:
        print('CO-SIMULATION RESULT:')
        print(f.read())


def _get_abs_and_percentage_values(unparsed_cell):
    return int(unparsed_cell.split('(')[0]), float(unparsed_cell.split('(')[1].replace('%', '').replace(')', ''))


def _parse_csim_results(hls_dir):
    """Parse C simulation results from tb_data/csim_results.log."""
    sim_file = _path('csim_results', hls_dir=hls_dir)
    if not os.path.isfile(sim_file):
        return None
    with open(sim_file) as f:
        return [[r for r in line.split()] for line in f.readlines()]


def _parse_rtl_cosim_results(hls_dir):
    """Parse RTL cosimulation results from tb_data/rtl_cosim_results.log."""
    sim_file = _path('rtl_cosim_results', hls_dir=hls_dir)
    if not os.path.isfile(sim_file):
        return None
    with open(sim_file) as f:
        return [[r for r in line.split()] for line in f.readlines()]


def _parse_csynthesis_report(sln_dir, solution, top_func_name):
    """Parse C synthesis XML report."""
    syn_file = _path('csynth_xml', sln_dir=sln_dir, solution=solution, top=top_func_name)
    if not os.path.isfile(syn_file):
        return None
    root = ET.parse(syn_file).getroot()
    c_synth_report = {}
    perf_node = root.find('./PerformanceEstimates')
    c_synth_report['TargetClockPeriod'] = root.find('./UserAssignments/TargetClockPeriod').text
    c_synth_report['EstimatedClockPeriod'] = perf_node.find('./SummaryOfTimingAnalysis/EstimatedClockPeriod').text
    c_synth_report['BestLatency'] = perf_node.find('./SummaryOfOverallLatency/Best-caseLatency').text
    c_synth_report['WorstLatency'] = perf_node.find('./SummaryOfOverallLatency/Worst-caseLatency').text
    c_synth_report['IntervalMin'] = perf_node.find('./SummaryOfOverallLatency/Interval-min').text
    c_synth_report['IntervalMax'] = perf_node.find('./SummaryOfOverallLatency/Interval-max').text
    area_node = root.find('./AreaEstimates')
    for child in area_node.find('./Resources'):
        if child.tag == 'DSP48E':
            child.tag = 'DSP'
        c_synth_report[child.tag] = child.text
    for child in area_node.find('./AvailableResources'):
        if child.tag == 'DSP48E':
            child.tag = 'DSP'
        c_synth_report['Available' + child.tag] = child.text
    return c_synth_report


def _parse_vivado_synth_report(hls_dir):
    """Parse Vivado synthesis report (vivado_synth.rpt)."""
    vivado_syn_file = _path('vivado_synth', hls_dir=hls_dir)
    if not os.path.isfile(vivado_syn_file):
        return None
    vivado_synth_rpt = {}
    with open(vivado_syn_file) as f:
        section = 0
        for line in f.readlines():
            match = re.match(r'^(\d)\.', line)
            if match:
                section = int(match.group(1))
            if '|' in line:
                if ('CLB LUTs' in line or 'Slice LUTs' in line) and section == VIVADO_SECTION_CLB:
                    vivado_synth_rpt['LUT'] = line.split('|')[VIVADO_COLUMN_INDEX].strip()
                elif ('CLB Registers' in line or 'Slice Registers' in line) and section == VIVADO_SECTION_CLB:
                    vivado_synth_rpt['FF'] = line.split('|')[VIVADO_COLUMN_INDEX].strip()
                elif 'Block RAM Tile' in line and section == VIVADO_SECTION_RAM:
                    vivado_synth_rpt['BRAM_18K'] = line.split('|')[VIVADO_COLUMN_INDEX].strip()
                elif 'URAM' in line and section == VIVADO_SECTION_RAM:
                    vivado_synth_rpt['URAM'] = line.split('|')[VIVADO_COLUMN_INDEX].strip()
                elif 'DSPs' in line and section == VIVADO_SECTION_DSP:
                    vivado_synth_rpt['DSP48E'] = line.split('|')[VIVADO_COLUMN_INDEX].strip()
    return vivado_synth_rpt


def _parse_transaction_file(sln_dir, solution, rtl, top_func_name):
    """Parse transaction file for detailed latency/interval stats. Returns dict to merge into CosimReport."""
    transaction_file = _path('transaction_file', sln_dir=sln_dir, solution=solution, rtl=rtl.lower(), top=top_func_name)
    if not os.path.isfile(transaction_file):
        return None
    cosim_transactions = {
        'InitiationInterval': {'max': 0, 'min': sys.maxsize, 'avg': 0.0},
        'Latency': {'max': 0, 'min': sys.maxsize, 'avg': 0.0},
    }
    with open(transaction_file) as f:
        i = 1
        for line in f.readlines():
            if re.search('transaction', line):
                result = line.split()
                if result[TX_INTERVAL_IDX] != 'x':
                    cosim_transactions['InitiationInterval']['min'] = min(
                        int(result[TX_INTERVAL_IDX]), cosim_transactions['InitiationInterval']['min']
                    )
                    cosim_transactions['InitiationInterval']['max'] = max(
                        int(result[TX_INTERVAL_IDX]), cosim_transactions['InitiationInterval']['max']
                    )
                    cosim_transactions['InitiationInterval']['avg'] += float(
                        (int(result[TX_INTERVAL_IDX]) - cosim_transactions['InitiationInterval']['avg']) / i
                    )
                cosim_transactions['Latency']['min'] = min(int(result[TX_LATENCY_IDX]), cosim_transactions['Latency']['min'])
                cosim_transactions['Latency']['max'] = max(int(result[TX_LATENCY_IDX]), cosim_transactions['Latency']['max'])
                cosim_transactions['Latency']['avg'] += float(
                    (int(result[TX_LATENCY_IDX]) - cosim_transactions['Latency']['avg']) / i
                )
                i += 1
    return {
        'LatencyMin': cosim_transactions['Latency']['min'],
        'LatencyMax': cosim_transactions['Latency']['max'],
        'LatencyAvg': cosim_transactions['Latency']['avg'],
        'IntervalMin': cosim_transactions['InitiationInterval']['min'],
        'IntervalMax': cosim_transactions['InitiationInterval']['max'],
        'IntervalAvg': cosim_transactions['InitiationInterval']['avg'],
    }


def _parse_implementation_report(hls_dir, is_vivado_accelerator):
    """Parse post-route utilization report."""
    util_rpt_path = 'util_rpt_system' if is_vivado_accelerator else 'util_rpt_vivado'
    post_route_util_file = _path(util_rpt_path, hls_dir=hls_dir)
    if not os.path.isfile(post_route_util_file):
        return None
    implementation_report = {}
    with open(post_route_util_file) as f:
        for line in f.readlines():
            if re.search(r'\(top\)', line):
                results = [_get_abs_and_percentage_values(elem) for elem in line.replace('|', '').split()[UTIL_SKIP_CELLS:]]
                implementation_report['TotLUTs'] = results[UTIL_COL_TOTLUTS][0]
                implementation_report['TotLUTs%'] = results[UTIL_COL_TOTLUTS][1]
                implementation_report['LogicLUTs'] = results[UTIL_COL_LOGICLUTS][0]
                implementation_report['LogicLUTs%'] = results[UTIL_COL_LOGICLUTS][1]
                implementation_report['LUTRAMs'] = results[UTIL_COL_LUTRAMS][0]
                implementation_report['LUTRAMs%'] = results[UTIL_COL_LUTRAMS][1]
                implementation_report['SRLs'] = results[UTIL_COL_SRLS][0]
                implementation_report['SRLs%'] = results[UTIL_COL_SRLS][1]
                implementation_report['FFs'] = results[UTIL_COL_FFS][0]
                implementation_report['FFs%'] = results[UTIL_COL_FFS][1]
                implementation_report['RAMB36s'] = results[UTIL_COL_RAMB36][0]
                implementation_report['RAMB36s%'] = results[UTIL_COL_RAMB36][1]
                implementation_report['RAMB18s'] = results[UTIL_COL_RAMB18][0]
                implementation_report['RAMB18s%'] = results[UTIL_COL_RAMB18][1]
                if len(results) == UTIL_COLS_WITH_URAM:
                    implementation_report['URAMs'] = results[UTIL_COL_URAM][0]
                    implementation_report['URAMs%'] = results[UTIL_COL_URAM][1]
                    implementation_report['DSPs'] = results[UTIL_COL_DSP][0]
                    implementation_report['DSPs%'] = results[UTIL_COL_DSP][1]
                else:
                    implementation_report['DSPs'] = results[UTIL_COL_DSP - 1][0]
                    implementation_report['DSPs%'] = results[UTIL_COL_DSP - 1][1]
                break
    return implementation_report if implementation_report else None


def _parse_timing_report(hls_dir, is_vivado_accelerator):
    """Parse post-route timing summary report."""
    timing_rpt_path = 'timing_summary_system' if is_vivado_accelerator else 'timing_summary_vivado'
    timing_report_file = _path(timing_rpt_path, hls_dir=hls_dir)
    if not os.path.isfile(timing_report_file):
        return None
    with open(timing_report_file) as f:
        while not re.search('WNS', next(f)):
            pass
        next(f)
        result = next(f).split()
    return {
        'WNS': float(result[TIMING_COL_WNS]),
        'TNS': float(result[TIMING_COL_TNS]),
        'WHS': float(result[TIMING_COL_WHS]),
        'THS': float(result[TIMING_COL_THS]),
        'WPWS': float(result[TIMING_COL_WPWS]),
        'TPWS': float(result[TIMING_COL_TPWS]),
    }


def _parse_power_report(hls_dir, is_vivado_accelerator):
    """Parse post-route power report."""
    power_rpt_path = 'power_rpt_system' if is_vivado_accelerator else 'power_rpt_vivado'
    power_report_file = _path(power_rpt_path, hls_dir=hls_dir)
    if not os.path.isfile(power_report_file):
        return None
    power_report = {}
    power_keys = {
        'Total On-Chip Power (W)': 'TotalOnChipPower',
        'Dynamic (W)': 'Dynamic',
        'Device Static (W)': 'Static',
    }
    with open(power_report_file) as f:
        for line in f:
            if '|' not in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
            label, value = parts[1], parts[2]
            for key_pattern, report_key in power_keys.items():
                if key_pattern in label:
                    power_report[report_key] = value
                    break
    return power_report if power_report else None


def parse_vivado_report(hls_dir):
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(_path('project_tcl', hls_dir=hls_dir)):
        prj_dir, top_func_name, backend_name = _parse_project_script(hls_dir)
    else:
        prj_dir, top_func_name, backend_name = None, None, 'vivado'

    if prj_dir is None or top_func_name is None:
        print('Unable to read project data. Exiting.')
        return

    sln_dir = _path('sln_dir', hls_dir=hls_dir, prj_dir=prj_dir)
    if not os.path.exists(sln_dir):
        print(f'Project {prj_dir} does not exist. Rerun `model_hls.build(...)`')
        return

    solutions = _find_solutions(sln_dir)
    if len(solutions) > 1:
        print(f'WARNING: Found {len(solutions)} solution(s) in {sln_dir}. Using the first solution.')

    is_vivado_accelerator = 'vivadoaccelerator' == backend_name

    solution = solutions[0]
    report = {}

    csim_results = _parse_csim_results(hls_dir)
    if csim_results is not None:
        report['CSimResults'] = csim_results

    cosim_results = _parse_rtl_cosim_results(hls_dir)
    if cosim_results is not None:
        report['CosimResults'] = cosim_results

    c_synth_report = _parse_csynthesis_report(sln_dir, solution, top_func_name)
    if c_synth_report is not None:
        report['CSynthesisReport'] = c_synth_report
    else:
        print('CSynthesis report not found.')

    vivado_synth_rpt = _parse_vivado_synth_report(hls_dir)
    if vivado_synth_rpt is not None:
        report['VivadoSynthReport'] = vivado_synth_rpt
    else:
        print('Vivado synthesis report not found.')

    transaction_data = _parse_transaction_file(sln_dir, solution, 'verilog', top_func_name)
    if transaction_data is not None:
        report['CosimReport'] = {
            'RTL': 'Verilog',
            'Status': 'PASS',
            **transaction_data,
        }
    else:
        print('Cosim report not found.')

    implementation_report = _parse_implementation_report(hls_dir, is_vivado_accelerator)
    if implementation_report is not None:
        report['ImplementationReport'] = implementation_report
    else:
        print('Implementation report not found.')

    timing_report = _parse_timing_report(hls_dir, is_vivado_accelerator)
    if timing_report is not None:
        report['TimingReport'] = timing_report
    else:
        print('Timing report not found.')

    power_report = _parse_power_report(hls_dir, is_vivado_accelerator)
    if power_report is not None:
        report['PowerReport'] = power_report
    else:
        print('Power report not found.')

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

_row_base_template = '        <tr><td>{row_title}</td><td>{{{row_key}}}</td>'


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
        return str(time_in_us) + ' \u00b5s'

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
        worst_latency = int(csynth_report['WorstLatency'])
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

    if 'PowerReport' in report_dict:
        body += make_header_template('Power report')
        perf_rows = {
            'Total On-Chip Power (W)': 'total',
            'Dynamic (W)': 'dynamic',
            'Device Static (W)': 'static',
        }
        body += make_table_template('Power', perf_rows)

        power_report = report_dict['PowerReport']

        params = {}
        params['total'] = power_report.get('TotalOnChipPower', 'N/A')
        params['dynamic'] = power_report.get('Dynamic', 'N/A')
        params['static'] = power_report.get('Static', 'N/A')

        body = body.format(**params)

    return body


def aggregate_graph_reports(graph_reports):
    """
    Aggregate the build results of each subgraph into a single dictionary.
    """

    if graph_reports is None or len(graph_reports) == 0:
        return {}

    keys_to_sum = ['BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM']
    first_subgraph = next(iter(graph_reports))
    reportChoice = 'CSynthesisReport' if 'VivadoSynthReport' not in graph_reports[first_subgraph] else 'VivadoSynthReport'
    base_report = graph_reports[first_subgraph][reportChoice]
    csynth_report = graph_reports[first_subgraph].get('CSynthesisReport', base_report)

    final_report = {
        'TargetClockPeriod': csynth_report.get('TargetClockPeriod', 'N/A'),
        'EstimatedClockPeriod': float(csynth_report.get('EstimatedClockPeriod', float('inf'))),
        'BestLatency': 'N/A',
        'WorstLatency': 'N/A',
    }

    final_report['AvailableBRAM_18K'] = csynth_report.get('AvailableBRAM_18K', 'N/A')
    final_report['AvailableDSP'] = csynth_report.get('AvailableDSP', 'N/A')
    final_report['AvailableFF'] = csynth_report.get('AvailableFF', 'N/A')
    final_report['AvailableLUT'] = csynth_report.get('AvailableLUT', 'N/A')
    final_report['AvailableURAM'] = csynth_report.get('AvailableURAM', 'N/A')

    for k in keys_to_sum:
        final_report[k] = float(base_report.get(k, '0'))

    for subgraph, data in graph_reports.items():
        if subgraph == first_subgraph:
            continue
        report = data.get(reportChoice, {})
        est_cp = float(report.get('EstimatedClockPeriod', float('inf')))
        if est_cp > final_report['EstimatedClockPeriod']:
            final_report['EstimatedClockPeriod'] = est_cp

        for k in keys_to_sum:
            final_report[k] += float(report.get(k, '0'))
            if k == 'DSP':
                final_report[k] += float(report.get('DSP48E', '0'))

    final_report['EstimatedClockPeriod'] = f'{final_report["EstimatedClockPeriod"]:.3f}'
    for k in keys_to_sum:
        final_report[k] = str(final_report[k])

    return {'StitchedDesignReport': final_report}
