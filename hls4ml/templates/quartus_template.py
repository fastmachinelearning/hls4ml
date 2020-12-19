import numpy as np
import math
import os
import ctypes
import copy
import platform
from bisect import bisect_left
import webbrowser
from calmjs.parse import es5
from calmjs.parse import asttypes
from tabulate import tabulate
import uuid
from ast import literal_eval

from hls4ml.templates.templates import Backend, cd
from hls4ml.model.hls_layers import IntegerPrecisionType, FixedPrecisionType


dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = {rfpad};
    static const unsigned bf_pad = {bfpad};

    static const unsigned reuse_factor = {reuse};
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

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

dense_function_template = 'nnet::dense{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
#conv1d_function_template = 'nnet::conv_1d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
#conv2d_function_template = 'nnet::conv_2d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
#pooling1d_function_template = 'nnet::pooling1d<{input_t}, {config}>({input}, {output});'
#pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {config}>({input}, {output});'
#merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
#resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'
#transpose_function_template = 'nnet::transpose{dim}<{input_t}, {config}>({input}, {output});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h']
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h']
#conv1d_include_list = ['nnet_utils/nnet_conv.h', 'nnet_utils/nnet_conv_large.h']
#conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_large.h']
activ_include_list = ['nnet_utils/nnet_activation.h']
#pooling_include_list = ['nnet_utils/nnet_pooling.h']
#merge_include_list = ['nnet_utils/nnet_merge.h']
#resize_include_list = ['nnet_utils/nnet_image.h']
#transpose_include_list = ['nnet_utils/nnet_array.h']

class QuartusBackend(Backend):
    def __init__(self):
        super(QuartusBackend, self).__init__('Quartus')
        self.register_templates('Dense'                  , dense_function_template, dense_config_template, dense_include_list)
        self.register_templates('BinaryDense'            , dense_function_template,       dense_config_template, dense_include_list)
        self.register_templates('BatchNormalization'     , batchnorm_function_template,   batchnorm_config_template, batchnorm_include_list)
        #self.register_templates('Conv1D'                 , conv1d_function_template,      [conv1d_config_template, conv_mult_config_template], conv1d_include_list)
        #self.register_templates('Conv2D'                 , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('Activation'             , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('ParametrizedActivation' , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('PReLU'                  , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('Softmax'                , activ_function_template,       softmax_config_template, activ_include_list)
        #self.register_templates('Pooling1D'              , pooling1d_function_template,   pooling1d_config_template, pooling_include_list)
        #self.register_templates('Pooling2D'              , pooling2d_function_template,   pooling2d_config_template, pooling_include_list)
        #self.register_templates('Merge'                  , merge_function_template,       merge_config_template, merge_include_list)
        #self.register_templates('Concatenate'            , merge_function_template,       concat_config_template, merge_include_list)
        #self.register_templates('Resize'                 , resize_function_template,      resize_config_template, resize_include_list)
        #self.register_templates('Transpose'              , transpose_function_template,   transpose_config_template, transpose_include_list)

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
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME
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
            print('WARNING: Invalid ReuseFactor={} in layer "{}". Using ReuseFactor={} instead. Valid ReuseFactor(s): {}.'
                .format(chosen_rf, layer.name, closest_rf, ','.join(map(str, valid_rf))))
            layer.reuse_factor = closest_rf

    def get_precision_string_backend(self, precision):
        if isinstance(precision, IntegerPrecisionType):
            typestring = 'ac_int<{width}, {signed}>'.format(width=precision.width, signed='false' if not precision.signed else 'true')
        elif isinstance(precision, FixedPrecisionType):
            args = [precision.width, precision.integer, 'false' if not precision.signed else 'true', precision.rounding_mode, precision.saturation_mode]
            args = ','.join([str(arg) for arg in args if arg is not None])
            typestring = 'ac_fixed<{args}>'.format(args=args)
        else:
            typestring = precision
        return typestring

    def set_strategy(self, layer):
        if layer.model.config.get_compression(layer):
            layer.set_attr('strategy', '_compressed')
        else:
            layer.model.config.backend.set_closest_reuse_factor(layer)
            layer.set_attr('strategy', '')

    def configure_weights(self, layer):
        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)
        layer.weights_original = copy.deepcopy(layer.weights['weight'])
        if not layer.model.config.get_compression(layer):
            self.gen_quartus_weight_array(layer)

    def bn_weight_fuse(self, model, node):
        dense_node = node.get_input_node()
        dense_weight = dense_node.weights_original
        dense_bias = dense_node.weights['bias']
        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        fused_weight = (bn_scale.data * dense_weight.data)
        fused_bias = bn_scale.data * dense_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        dense_node.weights['weight'].data = fused_weight
        dense_node.weights['bias'].data = fused_bias
        self.gen_quartus_weight_array(dense_node)


    def gen_quartus_weight_array(self, layer):
        block_factor = int((layer.attributes['n_in']*layer.attributes['n_out'])/layer.reuse_factor)
        bf_rounded = int(pow(2, np.ceil(np.log(block_factor)/np.log(2))))
        rf_rounded = int(pow(2, np.ceil(np.log(layer.reuse_factor)/np.log(2))))

        layer.weights['weight'].data = np.transpose(layer.weights['weight'].data).flatten()

        if(layer.attributes['n_in']*layer.attributes['n_out'] > 2048 and rf_rounded != layer.reuse_factor):
            layer.set_attr('rfpad', rf_rounded-layer.reuse_factor)
            layer.set_attr('bfpad', bf_rounded-block_factor)

            temp = np.empty([bf_rounded, rf_rounded])
            for i in range(rf_rounded):
                for j in range (bf_rounded):
                    if (i < layer.reuse_factor and j < block_factor):
                        w_index = i + layer.reuse_factor * j
                        temp[j][i] = layer.weights['weight'].data[w_index]
                    else:
                        temp[j][i] = 0
            layer.weights['weight'].data = temp.flatten()

        layer.weights['weight'].data_length = layer.weights['weight'].data.size
        return

    def validate_hls(self, config):
        if config.model_strategy.lower() == 'latency' and config.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
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

    def build(self, dir, prj_config, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, fpgasynth=False):
        """
        Low level function to build the system. Users should generally not call this function directly
        but instead use HLSModel.build(...)

        Args:
            dir (string):  The directory where the project is found
            prj_config (dict): The project configuration dictionary--note, not HLSConfig
            reset, optional: Whether to reset the system. (Currently ignored)
            synth, optional: Whether to run synthesis
            cosim, optional: Whether to run cosim (currently ignored)
            validation, optional: Whether to run validation (currently ignored)
            export, optional: Whether to export the project (currently ignored)
            fpgasynth, optional:  Whether to run fpga synthesis

        Errors raise exceptions
        """
        found = os.system('command -v i++ > /dev/null')
        if found != 0:
            raise Exception('Intel HLS installation not found. Make sure "i++" is on PATH.')

        top_func_name = prj_config["ProjectName"]

        # use a context manager for exception safety
        with cd(dir):
            if(synth):
                os.system('make {}-fpga'.format(top_func_name))
                os.system('./{}-fpga'.format(top_func_name))

            if(fpgasynth):
                found = os.system('command -v quartus_sh > /dev/null')
                if found != 0:
                    raise Exception('Quartus installation not found. Make sure "quartus_sh" is on PATH.')
                os.chdir(top_func_name + '-fpga.prj/quartus')
                os.system('quartus_sh --flow compile quartus_compile')

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
        return core_layers + dense_layers + norm_layers + activation_layers + qkeras_dense + skip_layers

    def get_pstring (self, width, intbits, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        decimal = width - intbits
        if decimal > 0:
            args = [width, intbits, 'false' if not signed else 'true', rounding_mode, saturation_mode]
            args = ', '.join([str(arg) for arg in args if arg is not None])
            return 'ac_fixed<{args}>'.format(args=args)
        else:
            return 'ac_int<{width}, {signed}>'.format(width=width, signed='false' if not signed else 'true')

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

        top_func_name = hls_config.get_project_name()
        prj_dir = top_func_name + '-fpga.prj'

        rpt_file = hls_dir + '/' + prj_dir + '/reports'
        if not os.path.exists(rpt_file):
            raise RuntimeError('Project {} does not exist. Make sure the project is built.'.format(prj_dir, hls_dir))

        report = self._find_reports(rpt_file)
        if output:
            print("HLS Resource Summary\n")
            print(tabulate(list(report.items())[0:10], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
            print("\n\nHLS Validation Summary\n")
            print(tabulate(list(report.items())[11:13], tablefmt='orgtbl', headers=['', '[Min, Max, Avg]']))
            if 'Clock' in report.keys():
                print("\n\nQuartus Synthesis Summary\n")
                print(tabulate(list(report.items())[13:], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
            else:
                print("Quartus compile data not found! To generate data run 'hls4ml build -l' or MODEL.build(synth=False, fpgasynth=True) if using API.")
        return report


    def read_report(self, hls_dir, prj_config, full_report=False, open_browser=False):
        """
        Low level function to print the report (and open browser). Users should generally not call this function directly
        but should use functions from the HLSModel.

        Args:
            dir (string):  The directory where the project is found
            prj_config (dict): The project configuration dictionary--note, not HLSConfig
            full_report, optional:  whether to have a full report (currently ignored)
            open_browser, optional:  whether to open a browser
        """
        if not os.path.exists(hls_dir):
            print('Path {} does not exist. Exiting.'.format(hls_dir))
            return

        try:
            top_func_name = prj_config['ProjectName']
        except Exception:
            print("prj_config is not a dictionary or it does not contain the ProjectName key")
            return

        prj_dir = top_func_name + '-fpga.prj'

        rpt_file = hls_dir + '/' + prj_dir + '/reports'
        if not os.path.exists(rpt_file):
            print('Project {} does not exist. Rerun "hls4ml build -p {} -b Quartus".'.format(prj_dir, hls_dir))
            return

        report = self._find_reports(rpt_file)
        print("HLS Resource Summary\n")
        print(tabulate(list(report.items())[0:10], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
        print("\n\nHLS Validation Summary\n")
        print(tabulate(list(report.items())[11:13], tablefmt='orgtbl', headers=['', '[Min, Max, Avg]']))
        if 'Clock' in report.keys():
            print("\n\nQuartus Synthesis Summary\n")
            print(tabulate(list(report.items())[13:], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
        else:
            print("Quartus compile data not found! To generate data run 'hls4ml build -l' or MODEL.build(synth=False, fpgasynth=True) if using API.")

        if open_browser:
            url = 'file:' + os.getcwd() + '/' + rpt_file + '/report.html'
            webbrowser.open(url)

    def _find_reports(self, sln_dir):
        def read_js_object(js_script):
            """
            Reads the JS input (js_script, a string), and return a dictionary of
            variables definded in the JS.
            """
            def visit(node):
                if isinstance(node, asttypes.Program):
                    d = {}
                    for child in node:
                        if not isinstance(child, asttypes.VarStatement):
                            raise ValueError("All statements should be var statements")
                        key, val = visit(child)
                        d[key] = val
                    return d
                elif isinstance(node, asttypes.VarStatement):
                    return visit(node.children()[0])
                elif isinstance(node, asttypes.VarDecl):
                    return (visit(node.identifier), visit(node.initializer))
                elif isinstance(node, asttypes.Object):
                    d = {}
                    for property in node:
                        key = visit(property.left)
                        value = visit(property.right)
                        d[key] = value
                    return d
                elif isinstance(node, asttypes.BinOp):
                    # simple constant folding
                    if node.op == '+':
                        if isinstance(node.left, asttypes.String) and isinstance(node.right, asttypes.String):
                            return visit(node.left) + visit(node.right)
                        elif isinstance(node.left, asttypes.Number) and isinstance(node.right, asttypes.Number):
                            return visit(node.left) + visit(node.right)
                        else:
                            raise ValueError("Cannot + on anything other than two literals")
                    else:
                        raise ValueError("Cannot do operator '%s'" % node.op)

                elif isinstance(node, asttypes.String) or isinstance(node, asttypes.Number):
                    return literal_eval(node.value)
                elif isinstance(node, asttypes.Array):
                    return [visit(x) for x in node]
                elif isinstance(node, asttypes.Null):
                    return None
                elif isinstance(node, asttypes.Boolean):
                    if str(node) == "false":
                        return False
                    else:
                        return True
                elif isinstance(node, asttypes.Identifier):
                    return node.value
                else:
                    raise Exception("Unhandled node: %r" % node)
            return visit(es5(js_script))

        def _read_quartus_file(filename, results):
            with open(filename) as dataFile:
                quartus_data = dataFile.read()
                quartus_data = read_js_object(quartus_data)

            if(quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock'] != "TBD"):
                results['Clock'] = quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock']
                results['Quartus ALM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['alm']
                results['Quartus REG'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['reg']
                results['Quartus DSP'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['dsp']
                results['Quartus RAM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['ram']
                results['Quartus MLAB'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['mlab']


        def _read_report_file(filename, results):
            with open(filename) as dataFile:
                report_data = dataFile.read()
                report_data = report_data[: report_data.rfind('var fileJSON')]
                report_data = read_js_object(report_data)
                results['HLS ALUT'], results['HLS FF'], results['HLS RAM'], results['HLS DSP'], results['HLS MLAB'] = report_data['areaJSON']['total']
                results['HLS ALUT percent'], results['HLS FF percent'], results['HLS RAM percent'], results['HLS DSP percent'], results['HLS MLAB percent'] = report_data['areaJSON']['total_percent']



        def _read_verification_file(filename, results):
            if os.path.isfile(filename):
                with open(filename) as dataFile:
                    verification_data = dataFile.read()
                    verification_data = read_js_object(verification_data)
                results['num_invocation'] = verification_data['verifJSON']['functions'][0]['data'][0]
                results['Latency'] = verification_data['verifJSON']['functions'][0]['data'][1].split(",")
                results['ii'] = verification_data['verifJSON']['functions'][0]['data'][2].split(",")
            else:
                print('Verification file not found. Run ./[projectname]-fpga to generate.')


        results = {}
        quartus_file = sln_dir + '/lib/quartus_data.js'
        report_file = sln_dir + '/lib/report_data.js'
        verification_file = sln_dir + '/lib/verification_data.js'
        _read_report_file(report_file, results)
        _read_verification_file(verification_file, results)
        _read_quartus_file(quartus_file, results)

        return results
