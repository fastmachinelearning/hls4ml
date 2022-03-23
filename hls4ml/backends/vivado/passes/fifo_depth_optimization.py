import json

from pyDigitalWaveTools.vcd.parser import VcdParser

import hls4ml
from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass


class FifoDepthOptimization(ConfigurableOptimizerPass, ModelOptimizerPass):
    def __init__(self):
        self.values = []

    def match(self, node):
        return True

    def _populate_values(self, name, data, depth):
        self.values.append({'name': name, 'data': [], 'max': 0, 'depth': 0})
        get_values = lambda x: int(x[1][1:], 2)
        self.values[-1]['data'] = [get_values(x) for x in data]
        self.values[-1]['max'] = max(self.values[-1]['data'])
        self.values[-1]['depth'] = int(depth[0][1][1:], 2)

    def transform(self, model):
        model.fifo_opt = True
        # use `large_fifo_depth = 0` to keep the default fifo depth
        profiling_fifo_depth = getattr(self, 'profiling_fifo_depth', 100000)

        # check axi-stream or io-stream, if not one the 2 exit
        if not(model.config.get_config_value('IOType') == 'io_stream'):
            raise Exception('To use this optimization you have to set `IOType` field to `io_stream` in the HLS config')

        # initialize all the fifos to `profiling_fifo_depth` so that they will be automatically implemented in BRAMs and so they will be
        # profiled

        if profiling_fifo_depth:

            for k, v in model.output_vars.items():
                if v == model.get_input_variables()[0] or v == model.get_output_variables()[0]:
                    continue
                v.pragma = (v.pragma[0], profiling_fifo_depth)

        model.write()
        model.build(reset=False, csim=True, synth=True, cosim=True, validation=False, export=False, vsynth=False)

        with open(
                model.config.get_output_dir() + '/' + model.config.get_project_name() + '_prj' + '/solution1/sim/verilog/fifo_opt.vcd') as vcd_file:
            vcd = VcdParser()
            vcd.parse(vcd_file)
            data = vcd.scope.toJson()

        n_elem = len(data['children'][0]['children'][0]['children'])
        for i in range(n_elem):
            name = data['children'][0]['children'][0]['children'][i]['name']
            data_p = data['children'][0]['children'][0]['children'][i]['children'][0]['data']
            depth = data['children'][0]['children'][0]['children'][i]['children'][1]['data']
            self._populate_values(name, data_p, depth)

        maxs = [{'name': i['name'], 'max': i['max'], 'depth': i['depth']} for i in self.values]

        with open(model.config.get_output_dir() + '/max_depth.json', 'w') as f:
            json.dump(maxs, f, indent=4)

        for k, v in model.output_vars.items():
            filtered_max = [x['max'] for x in maxs if v.cppname in x['name']]
            if len(filtered_max) == 0:
                continue
            if len(filtered_max) > 1:
                print('WARNING! Check names of FIFOs')
            v.pragma = (v.pragma[0], filtered_max[0] + 1)

        model.write()
        print('[hls4ml] - FIFO optimization completed')
        return False
