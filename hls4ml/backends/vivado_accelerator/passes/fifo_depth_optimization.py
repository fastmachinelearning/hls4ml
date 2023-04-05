from hls4ml.backends.vivado.passes.fifo_depth_optimization import (
    generate_max_depth_file,
    get_vcd_data,
    populate_values,
    set_big_fifos,
    set_fifo_depth,
)
from hls4ml.model.optimizer.optimizer import ConfigurableOptimizerPass, ModelOptimizerPass


class FifoDepthOptimization(ConfigurableOptimizerPass, ModelOptimizerPass):
    def __init__(self):
        self.values = []

    def transform(self, model):
        # use `large_fifo_depth = 0` to keep the default fifo depth
        profiling_fifo_depth = getattr(self, 'profiling_fifo_depth', 100_000)

        # check axi-stream or io-stream, if not one the 2 exit
        if not (
            model.config.get_config_value('IOType') == 'io_stream'
            or model.config.get_config_value('AcceleratorConfig')['Interface'] == 'axi_stream'
            or model.config.get_config_value('AcceleratorConfig')['Interface'] == 'axi_master'
        ):
            raise Exception(
                'To use this optimization you have to set `IOType` field to `io_stream` in the HLS config '
                'or `axi_stream` or `axi_master` in `AcceleratorConfig` interface field'
            )

        # initialize all the fifos to 10000 so that they will be automatically implemented in BRAMs and so they will be
        # profiled

        if profiling_fifo_depth:
            set_big_fifos(model.output_vars, profiling_fifo_depth)

        data = get_vcd_data(model)

        for i in range(1, len(data['children'][0]['children'][0]['children'])):
            # wrapper fifos
            populate_values(
                self.values,
                data['children'][0]['children'][0]['children'][i]['name'],
                data['children'][0]['children'][0]['children'][i]['children'][0]['data'],
                data['children'][0]['children'][0]['children'][i]['children'][1]['data'],
            )

        n_elem = len(data['children'][0]['children'][0]['children'][0]['children'])
        for i in range(n_elem):
            name = data['children'][0]['children'][0]['children'][0]['children'][i]['name']
            data_p = data['children'][0]['children'][0]['children'][0]['children'][i]['children'][0]['data']
            depth = data['children'][0]['children'][0]['children'][0]['children'][i]['children'][1]['data']
            populate_values(self.values, name, data_p, depth)

        maxs = [{'name': i['name'], 'max': i['max'], 'depth': i['depth']} for i in self.values]

        generate_max_depth_file(model, maxs)

        set_fifo_depth(model, maxs)

        inp = model.get_input_variables()[0]
        out = model.get_output_variables()[0]
        for x in maxs:
            if 'in_local' in x['name']:
                inp.pragma = (inp.pragma[0], x['max'] + 1)
            elif 'out_local' in x['name']:
                out.pragma = (out.pragma[0], x['max'] + 1)

        print('[hls4ml] - FIFO optimization completed')
        return False
