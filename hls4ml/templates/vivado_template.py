import numpy as np
import math
import os
import re
import ctypes
import platform
from bisect import bisect_left
import xml.etree.ElementTree as ET
import uuid

from hls4ml.templates.templates import Backend, cd
from hls4ml.model.hls_layers import IntegerPrecisionType, FixedPrecisionType

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
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t} table_t;
}};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {exp_table_t} exp_table_t;
    typedef {inv_table_t} inv_table_t;
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

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {new_height};
    static const unsigned new_width = {new_width};
}};\n"""

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned perm[3] = {{{perm_str}}};
}};\n"""

dense_function_template = 'nnet::dense_{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
conv1d_function_template = 'nnet::conv_1d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
conv2d_function_template = 'nnet::conv_2d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
pooling1d_function_template = 'nnet::pooling1d<{input_t}, {config}>({input}, {output});'
pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {config}>({input}, {output});'
merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'
transpose_function_template = 'nnet::transpose{dim}<{input_t}, {config}>({input}, {output});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_large.h']
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h']
conv1d_include_list = ['nnet_utils/nnet_conv.h', 'nnet_utils/nnet_conv_large.h']
conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_large.h']
activ_include_list = ['nnet_utils/nnet_activation.h']
pooling_include_list = ['nnet_utils/nnet_pooling.h']
merge_include_list = ['nnet_utils/nnet_merge.h']
resize_include_list = ['nnet_utils/nnet_image.h']
transpose_include_list = ['nnet_utils/nnet_array.h']

class VivadoBackend(Backend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self.register_templates('Dense', dense_function_template, dense_config_template, dense_include_list)
        self.register_templates('BinaryDense'            , dense_function_template,       dense_config_template, dense_include_list)
        self.register_templates('BatchNormalization'     , batchnorm_function_template,   batchnorm_config_template, batchnorm_include_list)
        self.register_templates('Conv1D'                 , conv1d_function_template,      [conv1d_config_template, conv_mult_config_template], conv1d_include_list)
        self.register_templates('Conv2D'                 , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('Activation'             , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('ParametrizedActivation' , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('PReLU'                  , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('Softmax'                , activ_function_template,       softmax_config_template, activ_include_list)
        self.register_templates('Pooling1D'              , pooling1d_function_template,   pooling1d_config_template, pooling_include_list)
        self.register_templates('Pooling2D'              , pooling2d_function_template,   pooling2d_config_template, pooling_include_list)
        self.register_templates('Merge'                  , merge_function_template,       merge_config_template, merge_include_list)
        self.register_templates('Concatenate'            , merge_function_template,       concat_config_template, merge_include_list)
        self.register_templates('Resize'                 , resize_function_template,      resize_config_template, resize_include_list)
        self.register_templates('Transpose'              , transpose_function_template,   transpose_config_template, transpose_include_list)

    def get_valid_reuse_factors(self, layer):
        n_in = 0
        n_out = 0
        if layer.__class__.__name__ == 'Dense':
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out')
        elif layer.__class__.__name__ == 'Conv1D':
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
        elif layer.__class__.__name__ == 'Conv2D':
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_height') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')

        max_rf = n_in * n_out
        valid_reuse_factors = []
        for rf in range(1, max_rf):
            _assert = self._check_conditions(n_in, n_out, rf)
            if _assert:
                valid_reuse_factors.append(rf)
        # Avoid using RF=1
        if valid_reuse_factors[0] == 1:
            valid_reuse_factors.pop(0)
        return valid_reuse_factors

    def _check_conditions(self, n_in, n_out, rf):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        #
        # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
        #
        _assert = (((multiplier_limit % n_out) == 0) or (rf >= n_in))
        _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
        #
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME OF VIVADO HLS
        #
        _assert = _assert and (((n_in * n_out) % rf) == 0)

        return _assert

    def get_closest_reuse_factor(self, valid_rf, chosen_rf):
        """
        Returns closest value to chosen_rf. valid_rf is sorted (obtained from get_valid_reuse_factors())
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(valid_rf, chosen_rf)
        if pos == 0:
            return valid_rf[0]
        if pos == len(valid_rf):
            return valid_rf[-1]
        before = valid_rf[pos - 1]
        after = valid_rf[pos]
        if after - chosen_rf < chosen_rf - before:
            return after
        else:
            return before

    def set_closest_reuse_factor(self, layer):
        valid_rf = self.get_valid_reuse_factors(layer)
        chosen_rf = layer.reuse_factor
        if chosen_rf not in valid_rf:
            closest_rf = self.get_closest_reuse_factor(valid_rf, chosen_rf)
            print('WARNING: Invalid ReuseFactor={} with "Resource" strategy in layer "{}". Using ReuseFactor={} instead. Valid ReuseFactor(s): {}.'
                .format(chosen_rf, layer.name, closest_rf, ','.join(map(str, valid_rf))))
            layer.reuse_factor = closest_rf

    def get_precision_string_backend(self, precision):
        if isinstance(precision, IntegerPrecisionType):
            typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not precision.signed else '', width=precision.width)
        elif isinstance(precision, FixedPrecisionType):
            args = [precision.width, precision.integer, precision.rounding_mode, precision.saturation_mode, precision.saturation_bits]
            args = ','.join([str(arg) for arg in args if arg is not None])
            typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not precision.signed else '', args=args)
        else:
            typestring = precision
        return typestring

    def set_strategy(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.model.config.backend.set_closest_reuse_factor(layer)
            if layer.model.config.get_compression(layer):
                layer.set_attr('strategy', 'compressed')
            else:
                layer.set_attr('strategy', 'large')
        else:
            layer.set_attr('strategy', 'latency')

    def configure_weights(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            if not layer.model.config.get_compression(layer):
                layer.weights['weight'].data = np.transpose(layer.weights['weight'].data)

    def bn_weight_fuse(self, model, node):
        dense_node = node.get_input_node()

        dense_weight = dense_node.weights['weight']
        dense_bias = dense_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        if dense_node.get_attr('strategy') != 'large':
            fused_weight = bn_scale.data * dense_weight.data
        else:
            fused_weight = (bn_scale.data * dense_weight.data.T).T

        fused_bias = bn_scale.data * dense_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        dense_node.weights['weight'].data = fused_weight
        dense_node.weights['bias'].data = fused_bias

    def validate_hls(self, config):
        use_resource = False
        if config.model_strategy.lower() == 'latency' and config.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            use_resource = True
        for layer_type, strategy in config.layer_type_strategy.items():
            if strategy.lower() == 'resource' and config.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer type {} set to "Resource", while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, strategy in config.layer_name_strategy.items():
            if strategy.lower() == 'resource' and config.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer {} set to "Resource", while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        for layer_type, compression in config.layer_type_compression.items():
            if compression and config.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer type {}, while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, compression in config.layer_name_compression.items():
            if compression and config.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer {}, while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        if use_resource:
            print('WARNING: Changing model strategy to "Resource"')
            config.model_strategy = 'Resource'

    def compile(self, model):
        libname = "{}".format(str(uuid.uuid4().hex))
        ret_val = os.system('bash build_lib.sh {}'.format(libname))
        if ret_val != 0:
            raise Exception('Failed to compile project "{}"'.format(model.config.get_project_name()))
        lib_name = 'firmware/{}.so'.format(libname)
        if model._top_function_lib is not None:

            if platform.system() == "Linux":
                dlclose_func = ctypes.CDLL('libdl.so').dlclose
            elif platform.system() == "Darwin":
                dlclose_func = ctypes.CDLL('libc.dylib').dlclose

            dlclose_func.argtypes = [ctypes.c_void_p]
            dlclose_func.restype = ctypes.c_int
            dlclose_func(model._top_function_lib._handle)
        model._top_function_lib = ctypes.cdll.LoadLibrary(lib_name)

    def build(self, dir, prj_config=None, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, fpgasynth=False):
        """
        Low level function to build the system. Users should generally not call this function directly
        but instead use HLSModel.build(...)

        Args:
            dir (string):  The directory where the project is found
            prj_config (dict), optional: The project configuration dictionary (currently ignored)
            reset, optional: Whether to reset the system.
            synth, optional: Whether to run synthesis
            cosim, optional: Whether to run cosim
            validation, optional: Whether to run validation
            export, optional: Whether to export the project
            fpgasynth, optional:  Whether to run fpga synthesis

        Errors raise exceptions
        """
        found = os.system('command -v vivado_hls > /dev/null')
        if found != 0:
            raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')

        # use a contex manager for exception safety
        with cd(dir):
            os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={fpgasynth}"'
                .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, fpgasynth=fpgasynth))

    def get_supportedlayers(self):
        #Define supported laers
        core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Reshape']
        dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
        conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
        pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
        norm_layers = ['BatchNormalization']
        activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
        merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate']
        qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D']
        qkeras_dense = ['QDense', 'QActivation']
        #Define layers to skip for conversion to HLS
        skip_layers = ['Dropout', 'Flatten']
        #All supported layers
        return core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    def get_pstring (self, width, intbits, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        decimal = width - intbits
        if decimal > 0:
            args = [width, intbits, rounding_mode, saturation_mode, saturation_bits]
            args = ', '.join([str(arg) for arg in args if arg is not None])
            return 'ap_{signed}fixed<{args}>'.format(signed='u' if not signed else '', args=args)
        else:
            return 'ap_{signed}int<{width}>'.format(signed='u' if not signed else '', width=width)

    def report_to_dict(self, hls_config, output=False):
        """
        Low level function to return the report as a dictionary. Users should generally not call this function directly
        but should use functions from the HLSModel.

        Args:
            dir (string):  The directory where the project is found
            hls_config (HLSConfig): The project configuration
            output, optional:  whether to pint a summary

        Returns:
            dict: the report dictionary

        Raises exceptions on errors

        """
        hls_dir = hls_config.get_output_dir()
        if not os.path.exists(hls_dir):
            raise RuntimeError('Path {} does not exist. Exiting.'.format(hls_dir))

        prj_dir = None
        top_func_name = None

        if os.path.isfile(hls_dir + '/build_prj.tcl'):
            prj_dir, top_func_name = self._parse_build_script(hls_dir + '/build_prj.tcl')

        if prj_dir is None or top_func_name is None:
            raise RuntimeError('Unable to read project data.')

        sln_dir = hls_dir + '/' + prj_dir
        if not os.path.exists(sln_dir):
            raise RuntimeError('Project {} does not exist. Make sure the project is built.'.format(prj_dir, hls_dir))

        solutions = self._find_solutions(sln_dir)
        if len(solutions) > 1:
            print('WARNING: Found {} solution(s) in {}. Using the first solution.'.format(len(solutions), sln_dir))

        report = {}

        sim_file = hls_dir + '/tb_data/csim_results.log'
        if os.path.isfile(sim_file):
            csim_results = []
            with open(sim_file, 'r') as f:
                for line in f.readlines():
                    csim_results.append([float(r) for r in line.split()])
            report['CSimResults'] = csim_results

        sim_file = hls_dir + '/tb_data/rtl_cosim_results.log'
        if os.path.isfile(sim_file):
            cosim_results = []
            with open(sim_file, 'r') as f:
                for line in f.readlines():
                    cosim_results.append([float(r) for r in line.split()])
            report['CosimResults'] = cosim_results

        syn_file = sln_dir + '/' + solutions[0] + '/syn/report/{}_csynth.xml'.format(top_func_name)
        if os.path.isfile(syn_file):
            root = ET.parse(syn_file).getroot()

            # Performance
            perf_node = root.find('./PerformanceEstimates')
            report['EstimatedClockPeriod'] = perf_node.find('./SummaryOfTimingAnalysis/EstimatedClockPeriod').text
            report['BestLatency'] = perf_node.find('./SummaryOfOverallLatency/Best-caseLatency').text
            report['WorstLatency'] = perf_node.find('./SummaryOfOverallLatency/Worst-caseLatency').text
            report['IntervalMin'] = perf_node.find('./SummaryOfOverallLatency/Interval-min').text
            report['IntervalMax'] = perf_node.find('./SummaryOfOverallLatency/Interval-max').text
            # Area
            area_node = root.find('./AreaEstimates')
            report["Resources"] = {}
            report["AvailableResources"] = {}
            for child in area_node.find('./Resources'):
                report["Resources"][child.tag] = child.text
            for child in area_node.find('./AvailableResources'):
                report["AvailableResources"][child.tag] = child.text
        else:
            print('Synthesis report not found.')

        cosim_file = sln_dir + '/' + solutions[0] + '/sim/report/{}_cosim.rpt'.format(top_func_name)
        if os.path.isfile(cosim_file):
            with open(cosim_file, 'r') as f:
                for line in f.readlines():
                    if re.search('VHDL', line) or re.search('Verilog', line):
                        result = line[1:].split() # [1:] skips the leading '|'
                        result = [res[:-1] if res[-1] == '|' else res for res in result]
                        # RTL, Status, Latency-min, Latency-avg, Latency-max, Interval-min, Interval-avg, Interval-max
                        if result[1] == 'NA':
                            continue
                        else:
                            report['CosimRTL'] = result[0]
                            report['CosimStatus'] = result[1]
                            report['CosimLatencyMin'] = result[2]
                            report['CosimLatencyMax'] = result[4]
                            report['CosimIntervalMin'] = result[5]
                            report['CosimIntervalMax'] = result[7]

        if output:
            self.read_report(hls_dir)
        return report

    def read_report(self, hls_dir, prj_config=None, full_report=False, open_browser=False):
        """
        Low level function to print the report (and open browser). Users should generally not call this function directly
        but should use functions from the HLSModel.

        Args:
            dir (string):  The directory where the project is found
            prj_config (dict), optional: The project configuration dictionary (currently ignored)
            full_report, optional:  whether to have a full report (currently ignored)
            open_browser, optional:  currently not supported (ignored)
        """
        if not os.path.exists(hls_dir):
            print('Path {} does not exist. Exiting.'.format(hls_dir))
            return

        prj_dir = None
        top_func_name = None

        if os.path.isfile(hls_dir + '/build_prj.tcl'):
            prj_dir, top_func_name = self._parse_build_script(hls_dir + '/build_prj.tcl')

        if prj_dir is None or top_func_name is None:
            print('Unable to read project data. Exiting.')
            return

        sln_dir = hls_dir + '/' + prj_dir
        if not os.path.exists(sln_dir):
            print('Project {} does not exist. Rerun "hls4ml build -p {}".'.format(prj_dir, hls_dir))
            return

        solutions = self._find_solutions(sln_dir)
        print('Found {} solution(s) in {}.'.format(len(solutions), sln_dir))

        for sln in solutions:
            print('Reports for solution "{}":\n'.format(sln))
            self._find_reports(sln_dir + '/' + sln, top_func_name, full_report)

    def _parse_build_script(self, script_path):
        prj_dir = None
        top_func_name = None

        with open(script_path, 'r') as f:
            for line in f.readlines():
                if 'open_project' in line:
                    prj_dir = line.split()[-1]
                elif 'set_top' in line:
                    top_func_name = line.split()[-1]

        return prj_dir, top_func_name

    def _find_solutions(self, sln_dir):
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

    def _find_reports(self, sln_dir, top_func_name, full_report=False):
        csim_file = sln_dir + '/csim/report/{}_csim.log'.format(top_func_name)
        if os.path.isfile(csim_file):
            self._show_csim_report(csim_file)
        else:
            print('C simulation report not found.')

        syn_file = sln_dir + '/syn/report/{}_csynth.rpt'.format(top_func_name)
        if os.path.isfile(syn_file):
            self._show_synth_report(syn_file, full_report)
        else:
            print('Synthesis report not found.')

        cosim_file = sln_dir + '/sim/report/{}_cosim.rpt'.format(top_func_name)
        if os.path.isfile(cosim_file):
            self._show_cosim_report(cosim_file)
        else:
            print('Co-simulation report not found.')

    @staticmethod
    def _show_csim_report(csim_file):
        with open(csim_file, 'r') as f:
            print('C SIMULATION RESULT:')
            print(f.read())

    @staticmethod
    def _show_synth_report(synth_file, full_report=False):
        with open(synth_file, 'r') as f:
            print('SYNTHESIS REPORT:')
            for line in f.readlines()[2:]:
                if not full_report and '* DSP48' in line:
                    break
                print(line, end = '')

    @staticmethod
    def _show_cosim_report(cosim_file):
        with open(cosim_file, 'r') as f:
            print('CO-SIMULATION RESULT:')
            print(f.read())
