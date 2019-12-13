
dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {index_t} index_t;
}};\n"""

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    typedef {bias_t} bias_t;
    typedef {scale_t} scale_t;
}};\n"""

conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned n_in = {n_in};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride = {stride};
    static const unsigned dilation = {dilation};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};\n"""

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
}};\n"""


conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};\n"""

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::{iotype};
}};\n"""

pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned pool_size = {pool_size};
    static const unsigned n_out = {n_out};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride = {stride};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
}};\n"""

pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned pool_height = {pool_height};
    static const unsigned pool_width = {pool_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
}};\n"""

merge_config_template = """struct config{index} : nnet::merge_config {{
    static const unsigned n_elem = {n_elem};
}};\n"""

concat_config_template = """struct config{index} : nnet::concat_config {{
    static const unsigned n_elem1_0 = {n_elem1_0};
    static const unsigned n_elem1_1 = {n_elem1_1};
    static const unsigned n_elem1_2 = {n_elem1_2};
    static const unsigned n_elem2_0 = {n_elem2_0};
    static const unsigned n_elem2_1 = {n_elem2_1};
    static const unsigned n_elem2_2 = {n_elem2_2};

    static const unsigned axis = {axis};
}};\n"""

garnet_config_template = """struct config{index} : nnet::garnet_config {{
    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {output_transform_weights_t} output_transform_weights_t;
    typedef {output_transform_biases_t} output_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;

    typedef {accum_t} accum_t;
    typedef {edge_weight_t} edge_weight_t;
    typedef {aggr_t} aggr_t;

    static const unsigned n_vertices = {n_vertices};
    static const unsigned n_in_features = {n_in_features};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_filters = {n_filters};
    static const unsigned n_propagate = {n_propagate};
    static const unsigned distance_bitwidth = 10;

    static const unsigned reuse_factor = {reuse};
}};
"""

config_templates = {
    'Dense'                  : dense_config_template,
    'BinaryDense'            : dense_config_template,
    'BatchNormalization'     : batchnorm_config_template,
    'Conv1D'                 : [conv1d_config_template, conv_mult_config_template],
    'Conv2D'                 : [conv2d_config_template, conv_mult_config_template],
    'Activation'             : activ_config_template,
    'ParametrizedActivation' : activ_config_template,
    'PReLU'                  : activ_config_template,
    'Pooling1D'              : pooling1d_config_template,
    'Pooling2D'              : pooling2d_config_template,
    'Merge'                  : merge_config_template,
    'Concatenate'            : concat_config_template,
    'GarNet'                 : garnet_config_template
}

dense_function_template = 'nnet::dense_{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
conv1d_function_template = 'nnet::conv_1d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
conv2d_function_template = 'nnet::conv_2d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
pooling1d_function_template = 'nnet::pooling1d<{input_t}, {config}>({input}, {output});'
pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {config}>({input}, {output});'
merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
garnet_function_template = 'nnet::garnet<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output}, {input_transform_weights}, {input_transform_biases}, {aggregator_distance_weights}, {aggregator_distance_biases}, {output_transform_weights}, {output_transform_biases});'

function_templates = {
    'Dense'                  : dense_function_template,
    'BinaryDense'            : dense_function_template,
    'BatchNormalization'     : batchnorm_function_template,
    'Conv1D'                 : conv1d_function_template,
    'Conv2D'                 : conv2d_function_template,
    'Activation'             : activ_function_template,
    'ParametrizedActivation' : param_activ_function_template,
    'PReLU'                  : param_activ_function_template,
    'Pooling1D'              : pooling1d_function_template,
    'Pooling2D'              : pooling2d_function_template,
    'Merge'                  : merge_function_template,
    'Concatenate'            : merge_function_template,
    'GarNet'                 : garnet_function_template
}

def get_config_template(kind):
    return config_templates.get(kind)

def get_function_template(kind):
    return function_templates.get(kind)

def register_templates(name, function_template, config_template):
    global function_templates
    global config_templates

    function_templates[name] = function_template
    config_templates[name] = config_template
