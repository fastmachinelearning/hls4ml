import os
import sys
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.catapult.catapult_types import CatapultArrayVariableConverter
from hls4ml.backends.fpga.fpga_types import ACTypeConverter, HLSTypeConverter
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Embedding,
    GarNet,
    GarNetStack,
    GlobalPooling1D,
    GlobalPooling2D,
    Layer,
    Pooling1D,
    Pooling2D,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    Softmax,
)
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType, PackedType
from hls4ml.report import parse_catapult_report
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.utils.fixed_point_utils import ceil_log2


class CatapultBackend(FPGABackend):
    def __init__(self):
        super().__init__('Catapult')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific attributes, recurrent_reuse_factor and static implementation
        rnn_layers = [
            SimpleRNN,
            LSTM,
            GRU,
        ]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('static', value_type=bool, default=True, description=descriptions.recurrent_static)
            )
            attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
            self.attribute_map[layer] = attrs

        # Add ParallelizationFactor to Conv1D/2D
        pf_layers = [
            Conv1D,
            Conv2D,
        ]

        for layer in pf_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('parallelization_factor', default=1, description=descriptions.conv_pf))
            self.attribute_map[layer] = attrs

        # Add ConvImplementation to Convolution+Pooling layers
        cnn_layers = [Conv1D, Conv2D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Pooling1D, Pooling2D]

        for layer in cnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(
                ChoiceAttribute(
                    'conv_implementation',
                    choices=['LineBuffer', 'Encoded'],
                    default='LineBuffer',
                    description=descriptions.conv_implementation,
                )
            )
            self.attribute_map[layer] = attrs

        sep_conv_layers = [SeparableConv1D, SeparableConv2D]
        for layer in sep_conv_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(TypeAttribute('dw_output', default=FixedPrecisionType(18, 8)))
            self.attribute_map[layer] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'catapult:inplace_stream_flatten',  # Inform downstream changed packsize in case of skipping flatten
            'catapult:reshape_stream',
            'catapult:clone_output',
            'catapult:insert_zero_padding_before_conv1d',
            'catapult:insert_zero_padding_before_conv2d',
            'catapult:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'catapult:merge_batch_norm_quantized_tanh',
            'catapult:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'catapult:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'catapult:remove_final_reshape',
            'catapult:optimize_pointwise_conv',
            'catapult:inplace_parallel_reshape',
            'catapult:inplace_stream_flatten',
            'catapult:skip_softmax',
            'catapult:fix_softmax_table_size',
            'catapult:process_fixed_point_quantizer_layer',
            'infer_precision_types',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        catapult_types = [
            'catapult:transform_types',
            'catapult:register_bram_weights',
            'catapult:generate_conv_streaming_instructions',
            'catapult:apply_resource_strategy',
            'catapult:generate_conv_im2col',
            'catapult:apply_winograd_kernel_transformation',
        ]
        catapult_types_flow = register_flow('specific_types', catapult_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'catapult:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['catapult:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'catapult:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[self._writer_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + catapult_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            catapult_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        tech='fpga',
        part='xcku115-flvb2104-2-i',
        asiclibs='nangate-45nm',
        fifo=None,
        clock_period=5,
        io_type='io_parallel',
    ):
        config = {}

        config['Technology'] = tech
        if tech == 'fpga':
            config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        else:
            config['ASICLibs'] = asiclibs if asiclibs is not None else 'nangate-45nm'
        config['ClockPeriod'] = clock_period
        config['FIFO'] = fifo
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        vhdl=False,
        verilog=True,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        ran_frame=5,
        sw_opt=False,
        power=False,
        da=False,
        bup=False,
    ):
        # print(f'ran_frame value: {ran_frame}')  # Add this line for debugging
        catapult_exe = 'catapult'
        if 'linux' in sys.platform:
            cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            found = os.system(cmd)
            if found != 0:
                catapult_exe = os.getenv('MGC_HOME') + '/bin/catapult'
                cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            found = os.system(cmd)
            if found != 0:
                catapult_exe = os.getenv('CATAPULT_HOME') + '/bin/catapult'
                cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            if found != 0:
                raise Exception('Catapult HLS installation not found. Make sure "catapult" is on PATH.')

        curr_dir = os.getcwd()
        # this execution moves into the hls4ml-generated "output_dir" and runs the build_prj.tcl script.
        os.chdir(model.config.get_output_dir())
        ccs_args = f'"reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation}'
        ccs_args += f' export={export} vsynth={vsynth} fifo_opt={fifo_opt} bitfile={bitfile} ran_frame={ran_frame}'
        ccs_args += f' sw_opt={sw_opt} power={power} da={da} vhdl={vhdl} verilog={verilog} bup={bup}"'
        ccs_invoke = catapult_exe + ' -product ultra -shell -f build_prj.tcl -eval \'set ::argv ' + ccs_args + '\''
        print(ccs_invoke)
        os.system(ccs_invoke)
        os.chdir(curr_dir)

        return parse_catapult_report(model.config.get_output_dir())

    def _validate_conv_strategy(self, layer):
        if layer.model.config.pipeline_style.lower() != 'dataflow':
            print(f'WARNING: Layer {layer.name} requires "dataflow" pipeline style. Switching to "dataflow" pipeline style.')
            layer.model.config.pipeline_style = 'dataflow'

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
        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))

    # TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_height * out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    def _set_pooling_accum_t(self, layer, pool_size):
        extra_bits = ceil_log2(pool_size)
        accum_t = layer.get_attr('accum_t')
        accum_t.precision.width += extra_bits * 2
        if isinstance(accum_t.precision, FixedPrecisionType):
            accum_t.precision.integer += extra_bits

    @layer_optimizer(Pooling1D)
    def init_pooling1d(self, layer):
        pool_size = layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling2D)
    def init_pooling2d(self, layer):
        pool_size = layer.get_attr('pool_height') * layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(GlobalPooling1D)
    def init_global_pooling1d(self, layer):
        pool_size = layer.get_attr('n_in')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(GlobalPooling2D)
    def init_global_pooling2d(self, layer):
        pool_size = layer.get_attr('in_height') * layer.get_attr('in_width')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(Softmax)
    def init_softmax(self, layer):
        if layer.model.config.get_config_value('IOType') == 'io_parallel':
            assert (
                len(layer.get_input_variable().shape) == 1
            ), 'Softmax with io_parallel strategy cannot be used on multidimensional tensors.'

    @layer_optimizer(Embedding)
    def init_embed(self, layer):
        if layer.attributes['n_in'] is None:
            raise Exception('Input length of Embedding layer must be specified.')

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GarNet)
    def init_garnet(self, layer):
        reuse_factor = layer.attributes['reuse_factor']

        var_converter = CatapultArrayVariableConverter(
            type_converter=HLSTypeConverter(precision_converter=ACTypeConverter())
        )

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
            out_pragma = ('partition', 'cyclic', partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)
