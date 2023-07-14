import numpy as np

from hls4ml.backends.fpga.fpga_types import APTypeConverter
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import GarNet, GarNetStack
from hls4ml.model.types import FixedPrecisionType

# GarNet templates

garnet_common_config_template = """
    static const unsigned n_vertices = {n_vertices};
    static const unsigned n_vertices_width = {n_vertices_width};
    static const unsigned n_in_features = {n_in_features};
    static const unsigned distance_width = {distance_width};
    static const unsigned output_collapse = {collapse_type};
    static const bool mean_by_nvert = {mean_by_nvert};

    typedef {norm_t} norm_t;
    typedef ap_fixed<{distance_width}, {distance_nint}, AP_TRN, AP_SAT> distance_t;
    typedef {edge_weight_t} edge_weight_t;
    typedef {edge_weight_aggr_t} edge_weight_aggr_t;
    typedef {aggr_t} aggr_t;
    typedef {output_t} output_t;

    static const unsigned reuse_factor = {reuse};
    static const unsigned log2_reuse_factor = {log2_reuse};
"""

garnet_config_template = """struct config{index} : nnet::garnet_config {{"""
garnet_config_template += garnet_common_config_template
garnet_config_template += """
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_weights_t} output_transform_weights_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_weights_t (&output_transform_weights)[{output_transform_weights_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index} base_t;
}};

const config{index}::input_transform_weights_t (&config{index}::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::input_transform_biases_t (&config{index}::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::aggregator_distance_weights_t (&config{index}::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::aggregator_distance_biases_t (&config{index}::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::output_transform_weights_t (&config{index}::output_transform_weights)[{output_transform_weights_size}] = {output_transform_weights};
const config{index}::output_transform_biases_t (&config{index}::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""  # noqa: E501

garnet_function_template = (
    'nnet::garnet{impl}<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'
)

garnet_include_list = ['nnet_utils/nnet_garnet.h']


class GarNetConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(GarNet)
        self.template = (garnet_config_template,)

    def get_transforms_config(self, node, params):
        params['n_in_features'] = node.attributes['n_in_features']
        params['n_propagate'] = node.attributes['n_propagate']
        params['n_aggregators'] = node.get_weights('aggregator_distance_biases').shape[0]
        params['n_out_features'] = node.get_weights('output_transform_biases').shape[0]

        for wname, weights in node.weights.items():
            params[wname] = weights.name
            params[f'{wname}_t'] = weights.type.name
            params[f'{wname}_size'] = weights.data_length

    def format(self, node):
        params = self._default_config_params(node)

        params['n_vertices'] = node.attributes['n_vertices']
        params['n_vertices_width'] = int(np.log2(params['n_vertices']))
        params['distance_width'] = 12
        params['distance_nint'] = min(4, params['distance_width'] - 6)  # this is tuned
        params['log2_reuse'] = int(np.log2(params['reuse']))

        # Define default precisions for various internal arrays (can be overridden from the config file)
        # We always give 10 digits for the subintegral part
        fwidth = 10
        # Integral precision for aggr_t depends on how large the temporary sum for weighed feature mean will be
        aggr_intw = max(params['log2_reuse'], params['n_vertices_width'] - params['log2_reuse']) + 3  # safety factor 2**3
        aggr_w = aggr_intw + fwidth
        # edge_weight_aggr_t does not need the safety factor
        ew_aggr_intw = aggr_intw - 3
        ew_aggr_w = ew_aggr_intw + fwidth
        # Integral precision for norm is fixed to 4
        norm_intw = 4
        norm_w = norm_intw + fwidth

        vspecs = [
            ('edge_weight', FixedPrecisionType(10, 0, signed=False)),
            ('edge_weight_aggr', FixedPrecisionType(ew_aggr_w, ew_aggr_intw, signed=False)),
            ('aggr', FixedPrecisionType(aggr_w, aggr_intw)),
            ('norm', FixedPrecisionType(norm_w, norm_intw, signed=False)),
        ]
        precision_converter = APTypeConverter()
        for vname, default_precision in vspecs:
            params[f'{vname}_t'], type_name = node.model.config.get_precision(node, var=vname)
            if type_name.endswith('default_t'):
                params[f'{vname}_t'] = precision_converter.convert(default_precision).definition_cpp()
            else:
                params[f'{vname}_t'] = precision_converter.convert(params[f'{vname}_t']).definition_cpp()
        params['output_t'] = node.get_output_variable().type.name

        if node.attributes['collapse'] in ['mean', 'max']:
            params['collapse_type'] = 'collapse_{}'.format(node.attributes['collapse'])
        else:
            params['collapse_type'] = 'no_collapse'

        params['mean_by_nvert'] = str(node.attributes['mean_by_nvert']).lower()

        self.get_transforms_config(node, params)

        return self.template[0].format(**params)


class GarNetFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(GarNet, include_header=garnet_include_list)
        self.template = garnet_function_template

    def format(self, node):
        params = self._default_function_params(node)

        data = node.get_input_variable(node.inputs[0])
        integer_input = node.get_input_variable(node.inputs[1])
        params['input_t'] = data.type.name
        params['input'] = data.name

        params['integer_input_t'] = integer_input.type.name
        params['nvtx'] = integer_input.name

        if node.ref_impl:
            params['impl'] = '_ref'
        else:
            params['impl'] = ''

        return self.template.format(**params)


# GarNetStack Templates

garnet_stack_base_config_template = """struct config{index}_base : nnet::garnet_config {{"""
garnet_stack_base_config_template += garnet_common_config_template
garnet_stack_base_config_template += """
    static const bool is_stack = true;

    typedef config{index}_base base_t;
}};

struct config{index} : config{index}_base {{
    static const unsigned n_sublayers = {n_sublayers};

    template<int L>
    struct sublayer_t : config{index}_base {{}};
}};

{sublayer_configs}
"""

garnet_stack_sublayer_config_template = """template<>
struct config{index}::sublayer_t<{il}> : config{index}_base {{
    static const unsigned n_in_features = {n_in_features};
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index}::sublayer_t<{next}> next_layer_t;
}};

const config{index}::sublayer_t<{il}>::input_transform_weights_t (&config{index}::sublayer_t<{il}>::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::sublayer_t<{il}>::input_transform_biases_t (&config{index}::sublayer_t<{il}>::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::sublayer_t<{il}>::aggregator_distance_weights_t (&config{index}::sublayer_t<{il}>::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::sublayer_t<{il}>::aggregator_distance_biases_t (&config{index}::sublayer_t<{il}>::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::sublayer_t<{il}>::output_transform_biases_t (&config{index}::sublayer_t<{il}>::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""  # noqa: E501

garnet_stack_config_template = (garnet_stack_base_config_template, garnet_stack_sublayer_config_template)
garnet_stack_function_template = (
    'nnet::garnet_stack<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'
)


class GarNetStackConfigTemplate(GarNetConfigTemplate):
    def __init__(self):
        super(GarNetConfigTemplate, self).__init__(GarNetStack)
        self.template = garnet_stack_config_template

    def get_transforms_config(self, node, params):
        _, sublayer_template = self.template

        params['n_sublayers'] = node.attributes['n_sublayers']
        params['n_in_features'] = node.attributes['n_in_features'][0]
        params['n_out_features'] = node.attributes['n_out_features'][-1]

        sublayer_configs = []
        for il in range(node.attributes['n_sublayers'] - 1, -1, -1):
            sub_params = {'index': node.index, 'il': il}

            for p in ['n_in_features', 'n_propagate', 'n_aggregators', 'n_out_features']:
                sub_params[p] = node.attributes[p][il]

            for wname, weights in node._sublayer_weights[il].items():
                sub_params[wname] = weights.name
                sub_params[f'{wname}_t'] = weights.type.name
                sub_params[f'{wname}_size'] = weights.data_length

            if il != node.attributes['n_sublayers'] - 1:
                sub_params['next'] = il + 1
            else:
                sub_params['next'] = 0

            sublayer_configs.append(sublayer_template.format(**sub_params))

        params['sublayer_configs'] = '\n'.join(sublayer_configs)


class GarNetStackFunctionTemplate(GarNetFunctionTemplate):
    def __init__(self):
        super(GarNetFunctionTemplate, self).__init__(GarNetStack, include_header=garnet_include_list)
        self.template = garnet_stack_function_template
