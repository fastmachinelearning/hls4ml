from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import GRU, LSTM, Bidirectional, Layer, TimeDistributed

# recurrent multiplication template

recr_mult_config_template_1 = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

recr_mult_config_template_2 = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {recurrent_bias_t.name} bias_t;
    typedef {recurrent_weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

recr_activ_config_template = """struct {type}_config{index}_recr : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

# LSTM + GRU templates

recr_config_template = """struct config{index} : nnet::{recr_type}_config {{
    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;  // Matrix
    typedef {recurrent_weight_t.name} recurrent_weight_t;  // Matrix
    typedef {bias_t.name} bias_t;  // Vector
    typedef {recurrent_bias_t.name} recurrent_bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_state = {n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const bool use_static = {static};
    static const bool pytorch_order = {pytorch};
}};\n"""

# Bidirectional templates

single_config_template = """struct config{index} : nnet::single_layer_config {{
    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;  // Matrix
    typedef {recurrent_weight_t.name} recurrent_weight_t;  // Matrix
    typedef {bias_t.name} bias_t;  // Vector
    typedef {recurrent_bias_t.name} recurrent_bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_state = {n_state};
    static const unsigned n_mult = {n_mult};
    static const bool pytorch_order = {pytorch};
}};\n"""

bidirectional_config_template = """struct config{index} : nnet::bidirectional_config {{
    typedef {forward_t} FORWARD_CONFIG;
    template<class x_T, class y_T, typename config_T, bool backward>
    using RNNfunc_forward = nnet::{forward_layer}<x_T, y_T, config_T, backward>;
    typedef {backward_t} BACKWARD_CONFIG;
    template<class x_T, class y_T, typename config_T, bool backward>
    using RNNfunc_backward = nnet::{backward_layer}<x_T, y_T, config_T, backward>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const bool use_static = {static};
    static const bool pytorch_order = {pytorch};
}};\n"""

recr_function_template = 'nnet::{recr_type}_stack<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {wr}, {b}, {br});'
recr_function_template_initial_states_lstm = 'nnet::{recr_type}_stack<{input_t}, {input2_t}, {input3_t}, {output_t}, {config}>({input}, {input2}, {input3}, {output}, {w}, {wr}, {b}, {br});'  # noqa: E501
recr_function_template_initial_states_gru = 'nnet::{recr_type}_stack<{input_t}, {input2_t}, {output_t}, {config}>({input}, {input2}, {output}, {w}, {wr}, {b}, {br});'  # noqa: E501

bidirectional_function_template = 'nnet::bidirectional_stack<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {wr}, {b}, {br}, {w_b}, {wr_b}, {b_b}, {br_b});'  # noqa: E501

recr_include_list = ['nnet_utils/nnet_recurrent.h']


class RecurrentConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((LSTM, GRU))
        self.template = recr_config_template
        self.act_template = activ_config_template
        self.recr_act_template = recr_activ_config_template
        self.mult1_template = recr_mult_config_template_1
        self.mult2_template = recr_mult_config_template_2

    def format(self, node):
        params = self._default_config_params(node)
        in_0, in_1 = map(str, node.get_input_variable().shape[:2])

        params['n_in'] = in_1
        params['n_sequence'] = in_0
        if node.get_attr('return_sequences'):
            out_0, out_1 = map(str, node.get_output_variable().shape[:2])
            params['n_sequence_out'] = out_0
            params['n_state'] = out_1
            params['n_out'] = out_1
        else:
            params['n_sequence_out'] = 1
            params['n_state'] = params['n_out'] = str(node.get_output_variable().shape[0])

        params['config_mult_t1'] = f'config{node.index}_1'
        params['config_mult_t2'] = f'config{node.index}_2'
        params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        params['strategy'] = node.get_attr('strategy')
        params['static'] = 'true' if node.attributes['static'] else 'false'
        params['pytorch'] = 'true' if node.get_attr('pytorch', False) else 'false'
        params['recr_type'] = node.class_name.lower()
        params['RECR_TYPE'] = node.class_name

        if node.class_name == 'LSTM':
            n_recr_mult = 4
        else:  # GRU
            n_recr_mult = 3

        recr_config = self.template.format(**params)

        act_params = self._default_config_params(node)
        recr_act_params = self._default_config_params(node)

        act_params['type'] = node.get_attr('activation')
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        if node.get_attr('return_sequences'):
            act_params['n_in'] = node.get_output_variable().shape[1]
            recr_act_params['n_in'] = node.get_output_variable().shape[1] * (n_recr_mult - 1)
        else:
            act_params['n_in'] = node.get_output_variable().shape[0]
            recr_act_params['n_in'] = node.get_output_variable().shape[0] * (n_recr_mult - 1)

        act_config = self.act_template.format(**act_params)
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        mult_params1 = self._default_config_params(node)
        mult_params2 = self._default_config_params(node)

        mult_params1['n_in'] = node.get_input_variable().shape[1]
        if node.get_attr('return_sequences'):
            mult_params1['n_out'] = node.get_output_variable().shape[1] * n_recr_mult
        else:
            mult_params1['n_out'] = node.get_output_variable().shape[0] * n_recr_mult
        mult_params1['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_params1['reuse'] = params['reuse']
        mult_params1['index'] = str(node.index) + '_1'
        mult_params1['nzeros'] = node.get_weights('weight').nzeros
        mult_params1['nonzeros'] = node.get_weights('weight').nonzeros

        namespace = params['namespace']

        if node.get_attr('strategy').lower() == 'latency':
            mult_params1['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if int(mult_params1['reuse_factor']) <= int(mult_params1['n_in']):
                mult_params1['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
            else:
                mult_params1['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
            # The 3rd case is never used
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params1['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_1'

        if node.get_attr('return_sequences'):
            mult_params2['n_in'] = node.get_output_variable().shape[1]
            mult_params2['n_out'] = node.get_output_variable().shape[1] * n_recr_mult
        else:
            mult_params2['n_in'] = node.get_output_variable().shape[0]
            mult_params2['n_out'] = node.get_output_variable().shape[0] * n_recr_mult
        mult_params2['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('recurrent_weight').type.precision
        )
        mult_params2['reuse'] = node.attributes['recurrent_reuse_factor']
        mult_params2['index'] = str(node.index) + '_2'
        mult_params2['nzeros'] = node.get_weights('recurrent_weight').nzeros
        mult_params2['nonzeros'] = node.get_weights('recurrent_weight').nonzeros

        if node.get_attr('strategy').lower() == 'latency':
            mult_params2['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if int(mult_params2['reuse_factor']) <= int(mult_params2['n_in']):
                mult_params2['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
            else:
                mult_params2['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
            # The 3rd case is never used
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params2['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_2'

        mult_config1 = self.mult1_template.format(**mult_params1)
        mult_config2 = self.mult2_template.format(**mult_params2)

        return mult_config1 + '\n' + mult_config2 + '\n' + recr_act_config + '\n' + act_config + '\n' + recr_config


class BidirectionalConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Bidirectional)
        self.template = bidirectional_config_template
        self.layer_template = single_config_template
        self.act_template = activ_config_template
        self.recr_act_template = recr_activ_config_template
        self.mult1_template = recr_mult_config_template_1
        self.mult2_template = recr_mult_config_template_2

    def format(self, node: Layer):

        # ----- Bidirectional Layer Config -----#
        params = self._default_config_params(node)

        params['n_in'] = node.get_input_variable().shape[1]
        params['n_sequence'] = node.get_input_variable().shape[0]
        if node.get_attr('return_sequences'):
            params['n_sequence_out'] = node.get_output_variable().shape[0]
        else:
            params['n_sequence_out'] = 1
        params['n_out'] = node.get_attr('n_out')
        params['strategy'] = node.get_attr('strategy')
        params['static'] = 'true' if node.attributes['static'] else 'false'
        params['pytorch'] = 'true' if node.get_attr('pytorch', False) else 'false'
        params['forward_t'] = f'config{node.index}_forward'
        params['backward_t'] = f'config{node.index}_backward'
        params['forward_layer'] = node.get_attr('forward_class_name').lower() + '_class'
        params['backward_layer'] = node.get_attr('backward_class_name').lower() + '_class'
        if node.attributes['static']:
            params['forward_layer'] += '_static'
            params['backward_layer'] += '_static'

        recr_config = self.template.format(**params)

        # ----- Forward and Backward Layers Config -----#
        result = ''
        for d in ['forward', 'backward']:
            if node.get_attr(f'{d}_class_name') == 'LSTM':
                n_recr_mult = 4
            else:  # GRU
                n_recr_mult = 3

            # ----- Layer Config -----#
            layer_params = self._default_config_params(node)
            layer_params['n_in'] = params['n_in']
            layer_params['pytorch'] = params['pytorch']
            layer_params['n_state'] = node.get_attr(f'{d}_n_states')
            layer_params['n_mult'] = 4
            if node.get_attr(f'{d}_class_name').lower() == 'gru':
                layer_params['n_mult'] = 3
            layer_params['config_mult_t1'] = f'config{node.index}_1_{d[0]}'
            layer_params['config_mult_t2'] = f'config{node.index}_2_{d[0]}'
            layer_params['recr_act_t'] = '{}_config{}_recr'.format(
                node.get_attr(f'{d}_recurrent_activation'), str(node.index) + f'_{d[0]}'
            )
            layer_params['act_t'] = '{}_config{}'.format(node.get_attr(f'{d}_activation'), str(node.index) + f'_{d[0]}')
            layer_params['RECR_TYPE'] = node.get_attr(f'{d}_class_name')

            layer_params['weight_t'] = layer_params[f'{d}_weight_t']
            layer_params['recurrent_weight_t'] = layer_params[f'{d}_recurrent_weight_t']
            layer_params['bias_t'] = layer_params[f'{d}_bias_t']
            layer_params['recurrent_bias_t'] = layer_params[f'{d}_recurrent_bias_t']
            layer_params['activation'] = layer_params[f'{d}_activation']
            layer_params['recurrent_activation'] = layer_params[f'{d}_recurrent_activation']

            layer_params['index'] = str(node.index) + f'_{d}'

            layer_config = self.layer_template.format(**layer_params)

            # ----- Activations Config -----#
            act_params = self._default_config_params(node)
            recr_act_params = self._default_config_params(node)

            act_params['type'] = node.get_attr(f'{d}_activation')
            recr_act_params['type'] = node.get_attr(f'{d}_recurrent_activation')
            act_params['index'] = str(node.index) + f'_{d[0]}'
            recr_act_params['index'] = str(node.index) + f'_{d[0]}'
            act_params['n_in'] = node.get_attr(f'{d}_n_states')
            recr_act_params['n_in'] = node.get_attr(f'{d}_n_states') * (n_recr_mult - 1)

            act_config = self.act_template.format(**act_params)
            recr_act_config = self.recr_act_template.format(**recr_act_params)

            # ----- Mult Config -----#
            mult_params1 = self._default_config_params(node)
            mult_params2 = self._default_config_params(node)

            mult_params1['n_in'] = node.get_input_variable().shape[1]
            mult_params1['n_out'] = node.get_attr(f'{d}_n_states') * n_recr_mult
            mult_params1['product_type'] = get_backend('vivado').product_type(
                node.get_input_variable().type.precision, node.get_weights(f'{d}_weight').type.precision
            )
            mult_params1['reuse'] = params['reuse']
            mult_params1['index'] = str(node.index) + f'_1_{d[0]}'
            mult_params1['nzeros'] = node.get_weights(f'{d}_weight').nzeros
            mult_params1['nonzeros'] = node.get_weights(f'{d}_weight').nonzeros

            mult_params1['bias_t'] = mult_params1[f'{d}_bias_t']
            mult_params1['weight_t'] = mult_params1[f'{d}_weight_t']
            mult_params2['recurrent_bias_t'] = mult_params2[f'{d}_recurrent_bias_t']
            mult_params2['recurrent_weight_t'] = mult_params2[f'{d}_recurrent_weight_t']

            namespace = params['namespace']

            if node.get_attr('strategy').lower() == 'latency':
                mult_params1['dense_function'] = 'nnet::DenseLatency'
            elif node.get_attr('strategy').lower() == 'resource':
                if int(mult_params1[f'{d}_reuse_factor']) <= int(mult_params1['n_in']):
                    mult_params1['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
                else:
                    mult_params1['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
                # The 3rd case is never used
            elif node.get_attr('strategy').lower() == 'resource_unrolled':
                mult_params1['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_1'

            mult_params2['n_in'] = node.get_attr(f'{d}_n_states')
            mult_params2['n_out'] = node.get_attr(f'{d}_n_states') * n_recr_mult
            mult_params2['product_type'] = get_backend('vivado').product_type(
                node.get_input_variable().type.precision, node.get_weights(f'{d}_recurrent_weight').type.precision
            )
            mult_params2['reuse'] = node.attributes[f'{d}_recurrent_reuse_factor']
            mult_params2['index'] = str(node.index) + f'_2_{d[0]}'
            mult_params2['nzeros'] = node.get_weights(f'{d}_recurrent_weight').nzeros
            mult_params2['nonzeros'] = node.get_weights(f'{d}_recurrent_weight').nonzeros

            if node.get_attr('strategy').lower() == 'latency':
                mult_params2['dense_function'] = 'nnet::DenseLatency'
            elif node.get_attr('strategy').lower() == 'resource':
                if int(mult_params2[f'{d}_reuse_factor']) <= int(mult_params2['n_in']):
                    mult_params2['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
                else:
                    mult_params2['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
                # The 3rd case is never used
            elif node.get_attr('strategy').lower() == 'resource_unrolled':
                mult_params2['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_2'

            mult_config1 = self.mult1_template.format(**mult_params1)
            mult_config2 = self.mult2_template.format(**mult_params2)

            result += (
                mult_config1 + '\n' + mult_config2 + '\n' + recr_act_config + '\n' + act_config + '\n' + layer_config + '\n'
            )

        return result + recr_config


class RecurrentFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((LSTM, GRU), include_header=recr_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        if params['pass_initial_states'] == 'true':
            params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
            params['input2'] = node.get_input_variable(node.inputs[1]).name
            if node.class_name == 'LSTM':
                params['input3'] = node.get_input_variable(node.inputs[2]).name
                params['input3_t'] = node.get_input_variable(node.inputs[2]).type.name

        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['wr'] = node.get_weights('recurrent_weight').name
        params['br'] = node.get_weights('recurrent_bias').name
        params['activation'] = node.get_attr('activation')
        params['recurrent_activation'] = node.get_attr('recurrent_activation')
        params['recr_type'] = node.class_name.lower()

        if params['pass_initial_states'] == 'true':
            if node.class_name == 'LSTM':
                template = recr_function_template_initial_states_lstm
            else:
                template = recr_function_template_initial_states_gru
        else:
            template = recr_function_template

        return template.format(**params)


class BidirectionalFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Bidirectional), include_header=recr_include_list)

    def format(self, node):
        params = self._default_function_params(node)

        # TO DO: Add initial states functions for pytorch settings

        params['w'] = node.get_weights('forward_weight').name
        params['b'] = node.get_weights('forward_bias').name
        params['wr'] = node.get_weights('forward_recurrent_weight').name
        params['br'] = node.get_weights('forward_recurrent_bias').name
        params['w_b'] = node.get_weights('backward_weight').name
        params['b_b'] = node.get_weights('backward_bias').name
        params['wr_b'] = node.get_weights('backward_recurrent_weight').name
        params['br_b'] = node.get_weights('backward_recurrent_bias').name

        template = bidirectional_function_template

        return template.format(**params)


time_distributed_config_template = """struct config{index} : nnet::time_distributed_config {{
    static const unsigned dim = {dim};

    static const unsigned n_time_steps = {n_time_steps};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
}};\n"""

time_distributed_loop_start_template = """for (int ts = 0; ts < config{index}::n_time_steps; ts++) {{
        {loop_mode}
        nnet::read_time_step_{dim}d<{input_t}, {config}>(ts, {input}, {output});"""

time_distributed_loop_end_template = """    nnet::write_time_step_{dim}d<{output_t}, {config}>(ts, {input}, {output});
    }}"""

time_distributed_include_list = ['nnet_utils/nnet_time_distributed.h']


class TimeDistributedConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(TimeDistributed)
        self.template = time_distributed_config_template

    def format(self, node):
        params = self._default_config_params(node)

        input_shape = node.get_input_variable().shape
        params['dim'] = len(input_shape)
        if node.name.endswith('_end'):
            params['dim'] += 1  # The input variable will be from the wrapped layer, without time dimension
        params['in_height'] = input_shape[-3] if params['dim'] == 4 else 1
        params['in_width'] = input_shape[-2] if params['dim'] >= 3 else 1
        params['n_chan'] = input_shape[-1]

        return self.template.format(**params)


class TimeDistributedFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((TimeDistributed), include_header=time_distributed_include_list)
        self.template_start = time_distributed_loop_start_template
        self.template_end = time_distributed_loop_end_template

    def format(self, node):
        params = self._default_function_params(node)

        input_shape = node.get_input_variable().shape
        params['dim'] = len(input_shape)
        if node.name.endswith('_end'):
            params['dim'] += 1  # The input variable will be from the wrapped layer, without time dimension

        loop_mode = node.get_attr('time_step_loop_parallelism')
        if loop_mode == 'unroll':
            params['loop_mode'] = '#pragma HLS UNROLL'
        elif loop_mode == 'pipeline':
            params['loop_mode'] = '#pragma HLS PIPELINE'
        else:
            params['loop_mode'] = ''

        if node.attributes['wrapped_layer'].name == node.name + '_end':
            return self.template_start.format(**params)
        else:
            return self.template_end.format(**params)
