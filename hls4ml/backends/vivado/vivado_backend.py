import numpy as np
import math
import os
import sys
from bisect import bisect_left
from queue import Queue
from collections.abc import Iterable

from hls4ml.model.types import FixedPrecisionType, NamedType, IntegerPrecisionType
from hls4ml.model.layers import Layer, Dense, BatchNormalization, Embedding, Conv1D, Conv2D, Conv2DBatchnorm, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Activation, ParametrizedActivation, PReLU, Softmax, Pooling1D, Pooling2D, GlobalPooling1D, GlobalPooling2D, ZeroPadding1D, ZeroPadding2D, Merge, Concatenate, Dot, Resize, Transpose, SimpleRNN, LSTM, GRU, GarNet, GarNetStack
from hls4ml.model.attributes import Attribute
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer, model_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.backends import FPGABackend
from hls4ml.backends.fpga.fpga_types import APTypeConverter, HLSTypeConverter, VivadoArrayVariableConverter
from hls4ml.report import parse_vivado_report

class VivadoBackend(FPGABackend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        extended_attrs = {
            SimpleRNN: [Attribute('recurrent_reuse_factor', default=1), Attribute('static', value_type=bool, default=True)],
            LSTM: [Attribute('recurrent_reuse_factor', default=1), Attribute('static', value_type=bool, default=True)],
            GRU: [Attribute('recurrent_reuse_factor', default=1), Attribute('static', value_type=bool, default=True)],
        }
        self.attribute_map.update(extended_attrs)

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'vivado:remove_final_reshape',
            'vivado:reshape_stream',
            'vivado:clone_output',
            'vivado:insert_zero_padding_before_conv1d',
            'vivado:insert_zero_padding_before_conv2d',
            'vivado:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'vivado:merge_batch_norm_quantized_tanh',
            'vivado:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'vivado:optimize_pointwise_conv',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'vivado:transform_types',
            'vivado:register_bram_weights',
            'vivado:generate_conv_streaming_instructions',
            'vivado:apply_resource_strategy',
        ]
        vivado_types_flow = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = [
            'make_stamp',
            'vivado:write_hls'
        ]
        self._writer_flow = register_flow('write', writer_passes, requires=['vivado:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'vivado:fifo_depth_optimization'
        ]

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=['vivado:ip'], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass for opt_pass in all_passes if opt_pass not in initializers + streaming_passes + quantization_passes + optimization_passes + vivado_types + templates + writer_passes + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            extras_flow = register_flow('extras', extras, requires=[init_flow], backend=self.name)
        else:
            extras_flow = None

        ip_flow_requirements = ['optimize', init_flow, streaming_flow, quantization_flow, optimization_flow, vivado_types_flow, extras_flow, template_flow]
        ip_flow_requirements = list(filter(None, ip_flow_requirements))

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(self, part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel'):
        config = {}

        config['Part'] = part if part is not None else 'xcku115-flvb2104-2-i'
        config['ClockPeriod'] = clock_period
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False, fifo_opt=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
        
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth} fifo_opt={fifo_opt}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth, fifo_opt=fifo_opt))
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        compression = layer.model.config.get_compression(layer)
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', NamedType('layer{}_index'.format(layer.index), index_t))

    #TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2: # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0,1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2: # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0,1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')
        
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Activation)
    def init_activation(self, layer):
        if 'table_t' not in layer.attributes:
            layer.set_attr('table_t', NamedType(name=layer.name + '_table_t', precision=FixedPrecisionType(width=18, integer=8)))
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)

    @layer_optimizer(Softmax)
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

        if layer.model.config.get_config_value('IOType') == 'io_parallel':
            assert len(layer.get_input_variable().shape) == 1, 'Softmax with io_parallel strategy cannot be used on multidimensional tensors.'

    @layer_optimizer(Embedding)
    def init_embed(self, layer):
        if layer.attributes['n_in'] is None:
           raise Exception('Input length of Embedding layer must be specified.')

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        index_t = IntegerPrecisionType(width=1, signed=False)

        if 'table_t' not in layer.attributes:
            layer.set_attr('table_t', FixedPrecisionType(width=18, integer=8))
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.weights['weight'].data = np.transpose(layer.weights['weight'].data)
            layer.weights['recurrent_weight'].data = np.transpose(layer.weights['recurrent_weight'].data)
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', index_t)

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        index_t = IntegerPrecisionType(width=1, signed=False)

        if 'table_t' not in layer.attributes:
            layer.set_attr('table_t', FixedPrecisionType(width=18, integer=8))
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.weights['weight'].data = np.transpose(layer.weights['weight'].data)
            layer.weights['recurrent_weight'].data = np.transpose(layer.weights['recurrent_weight'].data)
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', index_t)

    @layer_optimizer(GarNet)
    def init_garnet(self, layer):
        reuse_factor = layer.attributes['reuse_factor']
        
        var_converter = VivadoArrayVariableConverter(type_converter=HLSTypeConverter(precision_converter=APTypeConverter()))
        
        # A bit controversial but we are going to set the partitioning of the input here
        in_layer = layer.model.graph[layer.inputs[0]]
        in_var = layer.get_input_variable(layer.inputs[0])
        partition_factor = in_var.shape[1] * (in_var.shape[0] // reuse_factor)
        in_pragma = ('partition', 'cyclic', partition_factor)
        new_in_var = var_converter.convert(in_var, pragma=in_pragma)
        in_layer.set_attr(layer.inputs[0], new_in_var)

        if layer.attributes['collapse']:
            out_pragma = 'partition'
        else:
            partition_factor = layer._output_features * (layer.attributes['n_vertices'] // reuse_factor)
            out_pragma = ('partition', 'cyclic' , partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)
