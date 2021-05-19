import numpy as np
import math
import os
import sys
from bisect import bisect_left
from queue import Queue
from collections.abc import Iterable

from hls4ml.model.hls_types import IntegerPrecisionType
from hls4ml.backends.backend import custom_initializer, optimizer_pass, layer_optimizer
from hls4ml.backends import FPGABackend
from hls4ml.report import parse_vivado_report

dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {index_t} index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_width = {stride_width};
    static const unsigned dilation = {dilation};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_width> pixels[min_width];
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
}};
const ap_uint<config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
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
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
}};
const ap_uint<config{index}::filt_height * config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

sepconv_config_template = """struct config{index} {{
    typedef {depthwise_config} depthwise_config;
    typedef {pointwise_config} pointwise_config;
}};\n"""

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {exp_table_t.name} exp_table_t;
    typedef {inv_table_t.name} inv_table_t;
}};\n"""

pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_filt = {n_filt};
    static const unsigned pool_width = {pool_width};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride_width = {stride_width};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t.name} accum_t;
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
    typedef {accum_t.name} accum_t;
}};\n"""

global_pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride = {stride};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    typedef {accum_t.name} accum_t;
}};\n"""

global_pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t.name} accum_t;
}};\n"""

zeropad1d_config_template = """struct config{index} : nnet::padding1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_width = {out_width};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad2d_config_template = """struct config{index} : nnet::padding2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

merge_config_template = """struct config{index} : nnet::merge_config {{
    static const unsigned n_elem = {n_elem};
}};\n"""

dot_config_template = """struct config{index} : nnet::dot_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    typedef {accum_t.name} accum_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
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

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};
}};\n"""

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned perm[3] = {{{perm_str}}};
}};\n"""

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
"""

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
"""

garnet_stack_config_template = (garnet_stack_base_config_template, garnet_stack_sublayer_config_template)



dense_function_template = 'nnet::dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
conv1d_function_template = 'nnet::conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
conv2d_function_template = 'nnet::conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
sepconv1d_function_template = 'nnet::separable_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {d}, {p}, {z}, {b});'
sepconv2d_function_template = 'nnet::separable_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {d}, {p}, {z}, {b});'
depthconv2d_function_template = 'nnet::depthwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
pooling1d_function_template = 'nnet::pooling1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
global_pooling1d_function_template = 'nnet::global_pooling1d<{input_t}, {config}>({input}, {output});'
global_pooling2d_function_template = 'nnet::global_pooling2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'
transpose_function_template = 'nnet::transpose{dim}<{input_t}, {config}>({input}, {output});'
garnet_function_template = 'nnet::garnet{impl}<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'
garnet_stack_function_template = 'nnet::garnet_stack<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_stream.h']
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h', 'nnet_utils/nnet_batchnorm_stream.h']
conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']
conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']
sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']
activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']
pooling_include_list = ['nnet_utils/nnet_pooling.h', 'nnet_utils/nnet_pooling_stream.h']
padding_include_list = ['nnet_utils/nnet_padding.h', 'nnet_utils/nnet_padding_stream.h']
merge_include_list = ['nnet_utils/nnet_merge.h', 'nnet_utils/nnet_merge_stream.h']
resize_include_list = ['nnet_utils/nnet_image.h', 'nnet_utils/nnet_image_stream.h']
transpose_include_list = ['nnet_utils/nnet_array.h']
garnet_include_list = ['nnet_utils/nnet_garnet.h']

class VivadoBackend(FPGABackend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self.register_templates('Dense', dense_function_template, dense_config_template, dense_include_list)
        self.register_templates('BinaryDense'            , dense_function_template,       dense_config_template, dense_include_list)
        self.register_templates('BatchNormalization'     , batchnorm_function_template,   batchnorm_config_template, batchnorm_include_list)
        self.register_templates('Conv1D'                 , conv1d_function_template,      [conv1d_config_template, conv_mult_config_template], conv1d_include_list)
        self.register_templates('Conv2D'                 , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('Conv2DBatchnorm'        , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('SeparableConv1D'        , sepconv1d_function_template,   [sepconv_config_template, conv1d_config_template, conv1d_config_template, conv_mult_config_template, conv_mult_config_template], sepconv1d_include_list)
        self.register_templates('SeparableConv2D'        , sepconv2d_function_template,   [sepconv_config_template, conv2d_config_template, conv2d_config_template, conv_mult_config_template, conv_mult_config_template], sepconv2d_include_list)
        self.register_templates('DepthwiseConv2D'        , depthconv2d_function_template, [conv2d_config_template, conv_mult_config_template], sepconv2d_include_list)
        self.register_templates('Activation'             , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('ParametrizedActivation' , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('PReLU'                  , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('Softmax'                , activ_function_template,       softmax_config_template, activ_include_list)
        self.register_templates('Pooling1D'              , pooling1d_function_template,   pooling1d_config_template, pooling_include_list)
        self.register_templates('Pooling2D'              , pooling2d_function_template,   pooling2d_config_template, pooling_include_list)
        self.register_templates('GlobalPooling1D'        , global_pooling1d_function_template,   global_pooling1d_config_template, pooling_include_list)
        self.register_templates('GlobalPooling2D'        , global_pooling2d_function_template,   global_pooling2d_config_template, pooling_include_list)
        self.register_templates('ZeroPadding1D'          , zeropad1d_function_template,   zeropad1d_config_template, padding_include_list)
        self.register_templates('ZeroPadding2D'          , zeropad2d_function_template,   zeropad2d_config_template, padding_include_list)
        self.register_templates('Merge'                  , merge_function_template,       merge_config_template, merge_include_list)
        self.register_templates('Concatenate'            , merge_function_template,       concat_config_template, merge_include_list)
        self.register_templates('Dot'                    , merge_function_template,       dot_config_template, merge_include_list)
        self.register_templates('Resize'                 , resize_function_template,      resize_config_template, resize_include_list)
        self.register_templates('Transpose'              , transpose_function_template,   transpose_config_template, transpose_include_list)
        self.register_templates('GarNet'                 , garnet_function_template,      garnet_config_template, garnet_include_list)
        self.register_templates('GarNetStack'            , garnet_stack_function_template,garnet_stack_config_template, garnet_include_list)

    def create_initial_config(self, device='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel'):
        config = {}

        config['Device'] = device if device is not None else 'xcku115-flvb2104-2-i'
        config['ClockPeriod'] = clock_period
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
        
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    @layer_optimizer('Dense')
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        compression = layer.model.config.get_compression(layer)
        if layer.model.config.is_resource_strategy(layer):
            self.set_closest_reuse_factor(layer)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
                layer.weights['weight'].data = np.transpose(layer.weights['weight'].data)
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', index_t)

    @layer_optimizer('Conv1D')
    def init_conv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
            layer.weights['weight'].data = np.transpose(layer.weights['weight'].data, axes=[2, 0, 1]) #(W,C,F) => (F,W,C)
        else:
            layer.set_attr('strategy', 'latency')

    @layer_optimizer('SeparableConv1D')
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
            layer.weights['depthwise'].data = np.transpose(layer.weights['depthwise'].data, axes=[2, 0, 1]) #(W,C,F) => (F,W,C)
            layer.weights['pointwise'].data = np.transpose(layer.weights['pointwise'].data, axes=[2, 0, 1]) #(W,C,F) => (F,W,C)
        else:
            layer.set_attr('strategy', 'latency')

    @layer_optimizer('Conv2D')
    def init_conv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
            layer.weights['weight'].data = np.transpose(layer.weights['weight'].data, axes=[3, 0, 1, 2]) #(H,W,C,F) => (F,H,W,C)
        else:
            layer.set_attr('strategy', 'latency')

    @layer_optimizer('SeparableConv2D')
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
            layer.weights['depthwise'].data = np.transpose(layer.weights['depthwise'].data, axes=[3, 0, 1, 2]) #(H,W,C,F) => (F,H,W,C)
            layer.weights['pointwise'].data = np.transpose(layer.weights['pointwise'].data, axes=[3, 0, 1, 2]) #(H,W,C,F) => (F,H,W,C)
        else:
            layer.set_attr('strategy', 'latency')

    @layer_optimizer('DepthwiseConv2D')
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_closest_reuse_factor(layer)
            layer.weights['weight'].data = np.transpose(layer.weights['weight'].data, axes=[3, 0, 1, 2]) #(H,W,C,F) => (F,H,W,C)
        else:
            layer.set_attr('strategy', 'latency')

    @custom_initializer('Softmax')
    def init_softmax(self, layer):
        if 'exp_table_t' not in layer.attributes:
            layer.set_attr('exp_table_t', layer.get_attr('table_t'))
        if 'inv_table_t' not in layer.attributes:
            layer.set_attr('inv_table_t', layer.get_attr('table_t'))
        if layer.model.config.is_resource_strategy(layer):
            # 'resource' strategy = 'latency' for Softmax
            layer.set_attr('implementation', 'latency')
        else:
            layer.set_attr('implementation', layer.model.config.get_strategy(layer).lower())
