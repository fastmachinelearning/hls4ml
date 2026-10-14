import os

import yaml

from hls4ml.report.vivado_report import _parse_cosim_rpt, _parse_csynth_xml


class _ConfigLoader(yaml.SafeLoader):
    pass


# hls4ml_config.yml stores the source model with a custom tag such as !keras_model, which is not needed here
_ConfigLoader.add_multi_constructor('!', lambda loader, suffix, node: None)


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

    return report
