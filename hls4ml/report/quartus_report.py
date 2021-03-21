import os
import webbrowser
from calmjs.parse import es5
from calmjs.parse import asttypes
from tabulate import tabulate
from ast import literal_eval

def parse_quartus_report(hls_dir):
    """
    Parse a report from a given Quartus project as a dictionary.

    Args:
        hls_dir (string):  The directory where the project is found

    Returns:
        dict: the report dictionary

    Raises exceptions on errors

    """
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = _find_project_dir(hls_dir)

    rpt_dir = hls_dir + '/' + prj_dir + '/reports'
    if not os.path.exists(rpt_dir):
        print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
        return

    return _find_reports(rpt_dir)

def read_quartus_report(hls_dir, open_browser=False):
    """
    Parse and print the Quartus report to print the report. Optionally open a browser.

    Args:
        hls_dir (string):  The directory where the project is found
        open_browser, optional:  whether to open a browser
    """
    report = parse_quartus_report(hls_dir)

    print('HLS Resource Summary\n')
    print(tabulate(list(report.items())[0:10], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
    print('\n\nHLS Validation Summary\n')
    print(tabulate(list(report.items())[11:13], tablefmt='orgtbl', headers=['', '[Min, Max, Avg]']))
    if 'Clock' in report.keys():
        print('\n\nQuartus Synthesis Summary\n')
        print(tabulate(list(report.items())[13:], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
    else:
        print('Quartus compile data not found! To generate data run FPGA synthesis')

    if open_browser:
        url = 'file:' + os.getcwd() + '/' + _find_project_dir(hls_dir) + '/report.html'
        webbrowser.open(url)

def _find_project_dir(hls_dir):
    top_func_name = None

    with open(hls_dir + '/build_lib.sh', 'r') as f:
        for line in f.readlines():
            if 'PROJECT=' in line:
                top_func_name = line.split(sep='=')[-1].rstrip()

    return top_func_name + '-fpga.prj'

def _find_reports(rpt_dir):
    def read_js_object(js_script):
        """
        Reads the JS input (js_script, a string), and return a dictionary of
        variables definded in the JS.
        """
        def visit(node):
            if isinstance(node, asttypes.Program):
                d = {}
                for child in node:
                    if not isinstance(child, asttypes.VarStatement):
                        raise ValueError("All statements should be var statements")
                    key, val = visit(child)
                    d[key] = val
                return d
            elif isinstance(node, asttypes.VarStatement):
                return visit(node.children()[0])
            elif isinstance(node, asttypes.VarDecl):
                return (visit(node.identifier), visit(node.initializer))
            elif isinstance(node, asttypes.Object):
                d = {}
                for property in node:
                    key = visit(property.left)
                    value = visit(property.right)
                    d[key] = value
                return d
            elif isinstance(node, asttypes.BinOp):
                # simple constant folding
                if node.op == '+':
                    if isinstance(node.left, asttypes.String) and isinstance(node.right, asttypes.String):
                        return visit(node.left) + visit(node.right)
                    elif isinstance(node.left, asttypes.Number) and isinstance(node.right, asttypes.Number):
                        return visit(node.left) + visit(node.right)
                    else:
                        raise ValueError("Cannot + on anything other than two literals")
                else:
                    raise ValueError("Cannot do operator '%s'" % node.op)

            elif isinstance(node, asttypes.String) or isinstance(node, asttypes.Number):
                return literal_eval(node.value)
            elif isinstance(node, asttypes.Array):
                return [visit(x) for x in node]
            elif isinstance(node, asttypes.Null):
                return None
            elif isinstance(node, asttypes.Boolean):
                if str(node) == "false":
                    return False
                else:
                    return True
            elif isinstance(node, asttypes.Identifier):
                return node.value
            else:
                raise Exception("Unhandled node: %r" % node)
        return visit(es5(js_script))

    def _read_quartus_file(filename, results):
        with open(filename) as dataFile:
            quartus_data = dataFile.read()
            quartus_data = read_js_object(quartus_data)

        if(quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock'] != "TBD"):
            results['Clock'] = quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock']
            results['Quartus ALM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['alm']
            results['Quartus REG'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['reg']
            results['Quartus DSP'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['dsp']
            results['Quartus RAM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['ram']
            results['Quartus MLAB'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['mlab']


    def _read_report_file(filename, results):
        with open(filename) as dataFile:
            report_data = dataFile.read()
            report_data = report_data[: report_data.rfind('var fileJSON')]
            report_data = read_js_object(report_data)
            results['HLS ALUT'], results['HLS FF'], results['HLS RAM'], results['HLS DSP'], results['HLS MLAB'] = report_data['areaJSON']['total']
            results['HLS ALUT percent'], results['HLS FF percent'], results['HLS RAM percent'], results['HLS DSP percent'], results['HLS MLAB percent'] = report_data['areaJSON']['total_percent']



    def _read_verification_file(filename, results):
        if os.path.isfile(filename):
            with open(filename) as dataFile:
                verification_data = dataFile.read()
                verification_data = read_js_object(verification_data)
            results['num_invocation'] = verification_data['verifJSON']['functions'][0]['data'][0]
            results['Latency'] = verification_data['verifJSON']['functions'][0]['data'][1].split(",")
            results['ii'] = verification_data['verifJSON']['functions'][0]['data'][2].split(",")
        else:
            print('Verification file not found. Run ./[projectname]-fpga to generate.')


    results = {}
    quartus_file = rpt_dir + '/lib/quartus_data.js'
    report_file = rpt_dir + '/lib/report_data.js'
    verification_file = rpt_dir + '/lib/verification_data.js'
    _read_report_file(report_file, results)
    _read_verification_file(verification_file, results)
    _read_quartus_file(quartus_file, results)

    return results
