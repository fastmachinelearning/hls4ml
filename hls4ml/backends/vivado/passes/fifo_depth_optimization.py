import json

from pyDigitalWaveTools.vcd.parser import VcdParser

import hls4ml
from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass


class FifoDepthOptimization(ConfigurableOptimizerPass):
    def __init__(self):
        self.values = []

    def match(self, node):
        return True

    def _populate_values(self, name, data, depth):
        self.values.append({'name': name, 'data': [], 'max': 0, 'depth': 0})
        get_values = lambda x: int(x[1][1:], 2)
        self.values[-1]['data'] = [get_values(x) for x in data]
        self.values[-1]['max'] = max(self.values[-1]['data'])
        self.values[-1]['depth'] = int(depth[1:], 2)

    def transform(self, model, node):
        init_large_fifo = getattr(self, 'init_large_fifo', True)
        # cfg = model.config.config.copy()
        # hls_config = cfg['HLSConfig']

        if not model.config.config['HLSConfig']['Model']['FIFO_opt']:
            raise Exception('To use this optimization you have to set `FIFO_opt` field to True in the HLS config')

        # initialize all the fifos to 10000 so that they will be automatically implemented in BRAMs and so they will be
        # profiled

        if init_large_fifo:

            for k,v in model.output_vars.items():
                v.pragma[1] = 100000
            # note: it does not handle wrapper fifos

        with open(
                model.config.get_output_dir() + '/' + model.config.get_project_name() + '_prj' + '/solution1/sim/verilog/fifo_opt.vcd') as vcd_file:
            vcd = VcdParser()
            vcd.parse(vcd_file)
            data = vcd.scope.toJson()

        # wrapper fifos - useful only with VivadoAccelerator backend
        if model.config.get_config_value('Backend') == 'VivadoAccelerator':
            for i in range(1, len(data['children'][0]['children'][0]['children'])):
                self._populate_values(data['children'][0]['children'][0]['children'][i]['name'],
                                      data['children'][0]['children'][0]['children'][i]['children'][0]['data'],
                                      data['children'][0]['children'][0]['children'][i]['children'][1]['data'][0][1])

        # model layers fifos
        n_elem = len(data['children'][0]['children'][0]['children'][0]['children'])
        for i in range(n_elem):
            self._populate_values(data['children'][0]['children'][0]['children'][0]['children'][i]['name'],
                            data['children'][0]['children'][0]['children'][0]['children'][i]['children'][0]['data'],
                            data['children'][0]['children'][0]['children'][0]['children'][i]['children'][1]['data'][0][
                                1])

        maxs = [{'name': i['name'], 'max': i['max'], 'depth': i['depth']} for i in self.values]

        with open(model.config.get_output_dir() + '/max_depth.json', 'w') as f:
            json.dump(maxs, f, indent=4)

        for k, v in model.output_vars.items():
            filtered_max = [x['max'] for x in maxs if v.cppname in x['name']]
            if len(filtered_max) == 0:
                continue
            if len(filtered_max) > 1:
                print('WARNING! Check names of FIFOs')
            v.pragma[1] = filtered_max[0] + 1

        # the wrapper fifos have to be handled
        # for x in maxs:
        #     if 'in_local' in x['name']:
        #         new_config['LayerName']['in_local'] = {'StreamDepth': x['max'] + 1}
        #     elif 'out_local' in x['name']:
        #         new_config['LayerName']['out_local'] = {'StreamDepth': x['max'] + 1}

        print('[hls4ml] - FIFO optimization completed')
        return True
