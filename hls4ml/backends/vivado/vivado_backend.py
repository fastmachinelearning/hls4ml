import os
import sys
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.fpga.fpga_types import APTypeConverter, HLSTypeConverter
from hls4ml.backends.vivado.vivado_types import VivadoArrayVariableConverter
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Einsum,
    EinsumDense,
    Embedding,
    GarNet,
    GarNetStack,
    Layer,
    LayerNormalization,
    Pooling1D,
    Pooling2D,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    TimeDistributed,
)
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType, PackedType, RoundingMode, SaturationMode
from hls4ml.report import parse_vivado_report
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.utils.einsum_utils import parse_einsum


class VivadoBackend(FPGABackend):
    def __init__(self):
        super().__init__('Vivado')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific attributes, recurrent_reuse_factor and static implementation
        rnn_layers = [SimpleRNN, LSTM, GRU]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('static', value_type=bool, default=True, description=descriptions.recurrent_static)
            )
            attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
            self.attribute_map[layer] = attrs

        bidir_rnn_layers = [Bidirectional]
        for layer in bidir_rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('forward_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(ConfigurableAttribute('backward_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('forward_recurrent_reuse_factor', default=1, description=descriptions.reuse_factor)
            )
            attrs.append(
                ConfigurableAttribute('backward_recurrent_reuse_factor', default=1, description=descriptions.reuse_factor)
            )
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

        # Add LayerNorm attributes
        ln_layers = [LayerNormalization]
        for layer in ln_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('table_range_power2', default=0, description=descriptions.table_range_power2))
            attrs.append(ConfigurableAttribute('table_size', default=4096, description=descriptions.table_size))
            attrs.append(
                TypeAttribute(
                    'table',
                    default=FixedPrecisionType(
                        8, 5, signed=False, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.table_type,
                )
            )
            attrs.append(
                TypeAttribute(
                    'accum',
                    default=FixedPrecisionType(
                        14, 4, signed=True, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.accum_type,
                )
            )
            self.attribute_map[layer] = attrs

        # Add TimeStepLoopParallelism to TimeDistributed
        attrs = self.attribute_map.get(TimeDistributed, [])
        attrs.append(
            ChoiceAttribute(
                'time_step_loop_parallelism',
                choices=['Off', 'Unroll', 'Pipeline'],
                default='Off',
                description=descriptions.time_distributed_loop,
            )
        )
        self.attribute_map[TimeDistributed] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'vivado:inplace_stream_flatten',  # Inform downstream changed packsize in case of skipping flatten
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
            'vivado:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'vivado:remove_final_reshape',
            'vivado:optimize_pointwise_conv',
            'vivado:inplace_parallel_reshape',
            'vivado:inplace_stream_flatten',
            'vivado:skip_softmax',
            'vivado:fix_softmax_table_size',
            'infer_precision_types',
            'vivado:distributed_arithmetic_codegen',
            'vivado:distributed_arithmetic_einsum_codegen',
            'vivado:fuse_quantizer_into_d_a_layers',
            'vivado:process_fixed_point_quantizer_layer',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'vivado:transform_types',
            'vivado:register_bram_weights',
            'vivado:generate_conv_streaming_instructions',
            'vivado:apply_resource_strategy',
            'vivado:generate_conv_im2col',
            'vivado:generate_pointwise_conv1_d',
            'vivado:generate_unrolled_dense_resource',
            'vivado:set_pipeline_style',
            'vivado:d_a_latency_dense_template',
            'vivado:d_a_latency_conv_template',
        ]
        vivado_types_flow = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'vivado:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vivado:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'vivado:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=['vivado:ip'], backend=self.name)

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
            + vivado_types
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
            vivado_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='12.5%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        tb_output_stream='both',
        **_,
    ):
        """Create initial configuration of the Vivado backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 12.5%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            tb_output_stream (str, optional): Controls where to write the output. Options are 'stdout', 'file' and 'both'.
                Defaults to 'both'.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '12.5%'
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
            'TBOutputStream': tb_output_stream,
        }

        return config

    def augment_multigraph_writer(self, multi_model_config):
        """Augment the configuration of a multi-graph model."""
        """
        no return value
        """
        pass

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
    ):
        if 'linux' in sys.platform:
            found = os.system('command -v vivado_hls > /dev/null')
            if found != 0:
                raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')

        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        vivado_cmd = (
            f'vivado_hls -f build_prj.tcl "reset={reset} '
            f'csim={csim} '
            f'synth={synth} '
            f'cosim={cosim} '
            f'validation={validation} '
            f'export={export} '
            f'vsynth={vsynth} '
            f'fifo_opt={fifo_opt}"'
        )
        os.system(vivado_cmd)
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
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() == 'distributed_arithmetic':
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
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
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}".'
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            elif layer.model.config.get_config_value('IOType') == 'io_parallel':
                print(
                    f'Unrolled resource strategy cannot be combined with io_parallel in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() == 'distributed_arithmetic':
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]

        # Not overriding user parallelization factor, if already set and user has not specified a value
        user_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', None)
        layer_pf = layer.get_attr('parallelization_factor', None)
        chosen_pf = user_pf or layer_pf or 1
        if user_pf is not None and layer_pf is not None:
            if user_pf != layer_pf:
                warn(
                    f'For layer {layer.name}, parallelization factor of {layer_pf} is defined in the proxy-model, but is overridden by the user to {user_pf}.'  # noqa: E501
                )

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
        layer.set_attr('parallelization_factor', closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
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

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv1D)
    def init_depconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
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

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            elif layer.model.config.get_config_value('IOType') == 'io_parallel':
                print(
                    f'Unrolled resource strategy cannot be combined with io_parallel in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                layer.set_attr('strategy', 'resource_unrolled')
        elif layer.model.config.get_strategy(layer).lower() == 'distributed_arithmetic':
            rf = layer.get_attr('reuse_factor')
            if rf != 1:
                raise Exception(f'Layer {layer.name} has rf = {rf} != 1, but has strategy = "distributed_arithmetic".')
            layer.set_attr('strategy', 'distributed_arithmetic')
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]

        # Not overriding user parallelization factor, if already set and user has not specified a value
        user_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', None)
        layer_pf = layer.get_attr('parallelization_factor', None)
        chosen_pf = user_pf or layer_pf or 1
        if user_pf is not None and layer_pf is not None:
            if user_pf != layer_pf:
                warn(
                    f'For layer {layer.name}, parallelization factor of {layer_pf} is defined in the proxy-model, but is overridden by the user to {user_pf}.'  # noqa: E501
                )

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
        layer.set_attr('parallelization_factor', closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
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

    @layer_optimizer(Pooling1D)
    def init_pooling1d(self, layer):
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling2D)
    def init_pooling2d(self, layer):
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

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
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                self.set_closest_reuse_factor(
                    layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor', include_max_rf=False
                )
                layer.set_attr('strategy', 'resource_unrolled')
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
        elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
            use_resource_instead = False
            if layer.get_attr('reuse_factor', 1) == 1:
                print(
                    f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name}". '
                    'Using "resource" strategy instead.'
                )
                use_resource_instead = True
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            if use_resource_instead:
                self.set_closest_reuse_factor(layer, n_in, n_out)
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')
            else:
                self.set_closest_reuse_factor(layer, n_in, n_out, include_max_rf=False)
                self.set_closest_reuse_factor(
                    layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor', include_max_rf=False
                )
                layer.set_attr('strategy', 'resource_unrolled')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(TimeDistributed)
    def init_time_distributed(self, layer):
        loop_mode = layer.get_attr('time_step_loop_parallelism', 'off').lower()
        if loop_mode == 'unroll' and layer.model.config.get_config_value('IOType') == 'io_stream':
            warn(f'Cannot unroll time step loop in layer "{layer.name}" while using "io_stream".')
            loop_mode = 'off'
        layer.set_attr('time_step_loop_parallelism', loop_mode)

    @layer_optimizer(Bidirectional)
    def init_bidirectional(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)

        for i, d in enumerate(['forward', 'backward']):
            layer.set_attr(f'{d}_reuse_factor', reuse_factor)
            layer.set_attr(f'{d}_recurrent_reuse_factor', reuse_factor)

            if layer.model.config.is_resource_strategy(layer):
                n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)[i]
                self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor')
                self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor')
                layer.set_attr('strategy', 'resource')

            elif layer.model.config.get_strategy(layer).lower() == 'resource_unrolled':
                use_resource_instead = False
                if layer.get_attr('reuse_factor', 1) == 1:
                    print(
                        f'Unrolled resource strategy cannot be combined with reuse factor 1 in layer "{layer.name} ({d})". '
                        'Using "resource" strategy instead.'
                    )
                use_resource_instead = True

                n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)[i]
                if use_resource_instead:
                    self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor')
                    self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor')
                    layer.set_attr('strategy', 'resource')
                else:
                    self.set_closest_reuse_factor(layer, n_in, n_out, attribute=f'{d}_reuse_factor', include_max_rf=False)
                    self.set_closest_reuse_factor(
                        layer, n_in_recr, n_out_recr, attribute=f'{d}_recurrent_reuse_factor', include_max_rf=False
                    )
                    layer.set_attr('strategy', 'resource_unrolled')
            else:
                layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

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
            out_pragma = ('partition', 'cyclic', partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)

    @layer_optimizer(EinsumDense)
    def init_einsum_dense(self, layer: EinsumDense) -> None:
        kernel: np.ndarray = layer.attributes['weight_data']
        bias: np.ndarray | None = layer.attributes['bias_data']
        equation = layer.attributes['equation']
        inp_shape = layer.attributes['inp_shape']
        out_shape = layer.attributes['out_shape']

        kernel_shape = kernel.shape
        recipe = parse_einsum(equation, inp_shape, kernel_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp_tpose_idxs, ker_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        # Pre-transpose kernel (and bias) to save a transpose in cpp. Shouldn't matter for latency strategy though.
        # hls4ml dense acts like i,ij->j
        # parser assumes ij,j->i, so we need to transpose the kernel to match
        kernel = kernel.transpose(ker_tpose_idxs)
        kernel = kernel.reshape(recipe['I'], recipe['L1'], recipe['C']).transpose(0, 2, 1)

        def to_original_kernel(tkernel: np.ndarray) -> np.ndarray:
            _kernel = tkernel.transpose(0, 2, 1)
            _kernel = _kernel.reshape(tuple(kernel_shape[i] for i in ker_tpose_idxs))
            return _kernel.transpose(np.argsort(ker_tpose_idxs))

        # TODO: for weight in bram mode (resource), broadcasting bias here shall be avoided.
        if bias is not None:
            bias = np.broadcast_to(bias, out_shape).transpose(np.argsort(out_tpose_idxs))
        else:
            # The automatically created bias is just the last dimension of the output shape
            # Which is too small in general for einsum dense.
            # The transpose is just to match the shape in case of have real bias, no real effect.
            bias = np.zeros(out_shape).transpose(np.argsort(out_tpose_idxs))

        layer.attributes['weight_data'] = kernel
        layer.attributes['to_original_kernel'] = to_original_kernel
        layer.attributes['bias_data'] = bias
        layer.attributes['inp_tpose_idxs'] = inp_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']
        layer.attributes['n_free_data'] = recipe['L0']
        layer.attributes['n_free_kernel'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        layer.add_weights(compression=layer.model.config.get_compression(layer))
        layer.add_bias()

        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy in ('latency', 'resource', 'distributed_arithmetic'):
            layer.set_attr('strategy', strategy)
            return
        warn(f'Invalid strategy "{strategy}" for EinsumDense layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')

    @layer_optimizer(Einsum)
    def init_einsum(self, layer: Einsum) -> None:

        equation = layer.attributes['equation']
        inp0_shape = layer.attributes['inp0_shape']
        inp1_shape = layer.attributes['inp1_shape']

        recipe = parse_einsum(equation, inp0_shape, inp1_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp0_tpose_idxs, inp1_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        layer.attributes.update(recipe)
        layer.attributes['n_free0'] = recipe['L0']
        layer.attributes['n_free1'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']

        layer.attributes['inp0_tpose_idxs'] = inp0_tpose_idxs
        layer.attributes['inp1_tpose_idxs'] = inp1_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs

        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy.lower() == 'resource':
            layer.set_attr('strategy', 'resource')
            return
        if strategy.lower() in ('latency', 'distributed_arithmetic'):
            layer.set_attr('strategy', 'latency')
            return
        warn(f'Invalid strategy "{strategy}" for Einsum layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')
