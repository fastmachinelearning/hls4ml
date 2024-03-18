import json

from pyDigitalWaveTools.vcd.parser import VcdParser

from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass


def populate_values(values, name, data, depth):
    def get_values(x):
        return int(x[1][1:], 2)

    values.append({'name': name, 'data': [], 'max': 0, 'depth': 0})
    values[-1]['data'] = [get_values(x) for x in data]
    values[-1]['max'] = max(values[-1]['data'])
    values[-1]['depth'] = int(depth[0][1][1:], 2)
    return values


def set_big_fifos(vars_to_profile, profiling_fifo_depth):
    for v in vars_to_profile.values():
        if v.pragma:
            v.pragma = (v.pragma[0], profiling_fifo_depth)


def get_vcd_data(model):
    model.write()
    model.build(reset=False, csim=True, synth=True, cosim=True, validation=False, export=False, vsynth=False, fifo_opt=True)

    with open(
        model.config.get_output_dir()
        + '/'
        + model.config.get_project_name()
        + '_prj'
        + '/solution1/sim/verilog/fifo_opt.vcd'
    ) as vcd_file:
        vcd = VcdParser()
        vcd.parse(vcd_file)
        data = vcd.scope.toJson()
    return data


def generate_max_depth_file(model, maxs):
    with open(model.config.get_output_dir() + '/max_depth.json', 'w') as f:
        json.dump(maxs, f, indent=4)


def set_fifo_depth(model, maxs):
    for v in model.output_vars.values():
        if v.pragma:
            filtered_max = [x['max'] for x in maxs if v.name in x['name']]
            if len(filtered_max) == 0:
                continue
            if len(filtered_max) > 1:
                print('WARNING! Check names of FIFOs')
            v.pragma = (v.pragma[0], filtered_max[0] + 1)


class FifoDepthOptimization(ConfigurableOptimizerPass, ModelOptimizerPass):
    def __init__(self):
        self.values = []

    def transform(self, model):
        # use `large_fifo_depth = 0` to keep the default fifo depth
        profiling_fifo_depth = getattr(self, 'profiling_fifo_depth', 100_000)

        # check axi-stream or io-stream, if not one the 2 exit
        if not (model.config.get_config_value('IOType') == 'io_stream'):
            raise RuntimeError('To use this optimization you have to set `IOType` field to `io_stream` in the HLS config')

        # initialize all the fifos to `profiling_fifo_depth` so that they will be automatically implemented in BRAMs
        # and so they will be profiled
        if profiling_fifo_depth:
            vars_to_profile = {
                k: v
                for k, v in model.output_vars.items()
                if v != model.get_output_variables()[0] and v != model.get_input_variables()[0]
            }

            set_big_fifos(vars_to_profile, profiling_fifo_depth)

        data = get_vcd_data(model)

        if len(data['children']) == 0:
            print(
                "FIFO depth optimization found no FIFOs implemented using BRAMs in the design, no optimization is possible."
            )
            print("Consider increasing profiling_fifo_depth.")
            return False

        n_elem = len(data['children'][0]['children'][0]['children'])
        for i in range(n_elem):
            name = data['children'][0]['children'][0]['children'][i]['name']
            data_p = data['children'][0]['children'][0]['children'][i]['children'][0]['data']
            depth = data['children'][0]['children'][0]['children'][i]['children'][1]['data']
            populate_values(self.values, name, data_p, depth)

        maxs = [{'name': i['name'], 'max': i['max'], 'depth': i['depth']} for i in self.values]

        generate_max_depth_file(model, maxs)

        set_fifo_depth(model, maxs)

        print('[hls4ml] - FIFO optimization completed')
        return False
