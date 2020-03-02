from __future__ import print_function
import os
import re
import xml.etree.ElementTree as ET

def read_vivado_report(hls_dir, full_report=False):
    if not os.path.exists(hls_dir):
        print('Path {} does not exist. Exiting.'.format(hls_dir))
        return

    prj_dir = None
    top_func_name = None

    if os.path.isfile(hls_dir + '/build_prj.tcl'):
        prj_dir, top_func_name = _parse_build_script(hls_dir + '/build_prj.tcl')

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

def _parse_build_script(script_path):
    prj_dir = None
    top_func_name = None

    with open(script_path, 'r') as f:
        for line in f.readlines():
            if 'open_project' in line:
                prj_dir = line.split()[-1]
            elif 'set_top' in line:
                top_func_name = line.split()[-1]
    
    return prj_dir, top_func_name

def _find_solutions(sln_dir):
    solutions = []

    if os.path.isfile(sln_dir + '/vivado_hls.app'):
        with open(sln_dir + '/vivado_hls.app') as f:
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
