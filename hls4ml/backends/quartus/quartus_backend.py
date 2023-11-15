import os
from contextlib import contextmanager

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.model.attributes import ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import GRU, LSTM, Activation, Conv1D, Conv2D, Dense, Embedding, Layer, SimpleRNN, Softmax
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType
from hls4ml.report import parse_quartus_report


@contextmanager
def chdir(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


class QuartusBackend(FPGABackend):
    def __init__(self):
        super().__init__('Quartus')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific recurrent_reuse_factor attribute
        rnn_layers = [
            SimpleRNN,
            LSTM,
            GRU,
        ]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1))
            attrs.append(ConfigurableAttribute('table_size', default=1024))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8)))
            self.attribute_map[layer] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = ['quartus:reshape_stream', 'quartus:clone_output']
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quartus_types = [
            'quartus:transform_types',
            'quartus:register_bram_weights',
            'quartus:apply_resource_strategy',
            'quartus:apply_winograd_kernel_transformation',
        ]
        quartus_types_flow = register_flow('specific_types', quartus_types, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'quartus:merge_batch_norm_quantized_tanh',
            'quartus:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'quartus:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'quartus:remove_final_reshape',
            'quartus:optimize_pointwise_conv',
            'quartus:inplace_parallel_reshape',
            'quartus:inplace_stream_flatten',
            'quartus:skip_softmax',
            'quartus:fix_softmax_table_size',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'quartus:write_hls']

        self._writer_flow = register_flow('write', writer_passes, requires=['quartus:ip'], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quartus_types
            + quantization_passes
            + templates
            + optimization_passes
            + writer_passes
        ]

        if len(extras) > 0:
            extras_flow = register_flow('extras', extras, requires=[init_flow], backend=self.name)
        else:
            extras_flow = None

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            quartus_types_flow,
            extras_flow,
            template_flow,
        ]
        ip_flow_requirements = list(filter(None, ip_flow_requirements))

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(self, part='Arria10', clock_period=5, io_type='io_parallel'):
        config = {}

        config['Part'] = part if part is not None else 'Arria10'
        config['ClockPeriod'] = clock_period
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def build(self, model, synth=True, fpgasynth=False, log_level=1, cont_if_large_area=False):
        """
        Builds the project using Intel HLS compiler.

        Args:
            model (ModelGraph): The model to build
            synth, optional: Whether to run HLS synthesis
            fpgasynth, optional:  Whether to run FPGA synthesis (Quartus Compile)
            log_level, optional: Logging level to be displayed during HLS synthesis (0, 1, 2)
            cont_if_large_area: Instruct the HLS compiler to continue synthesis if the estimated resource usage exceeds
                device resources
        Errors raise exceptions
        """

        # Check software needed is present
        found = os.system('command -v i++ > /dev/null')
        if found != 0:
            raise Exception('Intel HLS installation not found. Make sure "i++" is on PATH.')

        if fpgasynth:
            if fpgasynth and not synth:
                raise Exception('HLS Synthesis needs to be run before FPGA synthesis')
            found = os.system('command -v quartus_sh > /dev/null')
            if found != 0:
                raise Exception('Quartus installation not found. Make sure "quartus_sh" is on PATH.')

        with chdir(model.config.get_output_dir()):
            if synth:
                quartus_compile = 'QUARTUS_COMPILE=--quartus-compile' if fpgasynth else ''
                cont_synth = 'CONT_IF_LARGE_AREA=--dont-error-if-large-area-est' if cont_if_large_area else ''
                log_1 = 'LOGGING_1=-v ' if log_level >= 1 else ''
                log_2 = 'LOGGING_2=-v ' if log_level >= 2 else ''
                os.system(f'make {model.config.get_project_name()}-fpga {log_1} {log_2} {cont_synth} {quartus_compile}')

                # If running i++ through a container, such a singularity, this command will throw an exception, because the
                # host OS doesn't have access to HLS simulation tools. To avoid the exception, shell into the container
                # (e.g. singularity shell ....) and then execute the following command manually
                # This command simply tests the IP using a simulation tool and obtains the latency and initiation interval
                os.system(f'./{model.config.get_project_name()}-fpga')

        return parse_quartus_report(model.config.get_output_dir())

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)

        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)

        if layer.model.config.get_compression(layer):
            layer.set_attr('strategy', 'compressed')
        else:
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            layer.set_attr('strategy', 'resource')

        if layer.model.config.is_resource_strategy(layer):
            if layer.model.config.get_compression(layer):
                index_t = layer.get_weights('weight').type.index_precision

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))

    @layer_optimizer(Activation)
    def init_activation(self, layer):
        if layer.get_attr('activation') == 'tanh':
            layer.set_attr('activation', 'dense_tanh')
        if layer.get_attr('recurrent_activation') == 'tanh':
            layer.set_attr('recurrent_activation', 'dense_tanh')

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

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        # Dense multiplication properties
        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)

        index_t = IntegerPrecisionType(width=1, signed=False)
        layer.set_attr('index_t', index_t)

        if 'table_t' not in layer.attributes:
            layer.set_attr(
                'table_t', NamedType(name=layer.name + '_table_t', precision=FixedPrecisionType(width=18, integer=8))
            )
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)
        if True:  # layer.model.config.is_resource_strategy(layer): ... Quartus only supports Dense resource multiplication
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')

        layer.set_attr('index_t', index_t)

    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        # This can happen if we assign weights of Dense layer to 1x1 Conv1D
        if len(layer.weights['weight'].data.shape) == 2:
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        # Dense matrix multiply properties
        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)

        # Reuse and parallelization factors
        layer.set_attr('strategy', 'resource')
        n_in, n_out = self.get_layer_mult_size(layer)
        self.set_target_reuse_factor(layer)
        self.set_closest_reuse_factor(layer, n_in, n_out)
        layer.set_attr('parallelization', layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1))

        # impl_filt_width determines the filter size post-Winograd transformation
        layer.set_attr('impl_filt_width', layer.get_attr('filt_width'))

        # Implementation:
        # - combination - at compile-time, the decision between Winograd and im2col is made
        # - im2col - specifically use im2col
        # - Winograd - use Winograd, if possible
        layer.set_attr('implementation', layer.model.config.get_layer_config_value(layer, 'Implementation', 'combination'))

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Not used yet as there is no codegen implementation of CNNs for Quartus backend

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        # This can happen if we assign weights of Dense layer to 1x1 Conv2D
        if len(layer.weights['weight'].data.shape) == 2:
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        # Dense matrix multiply properties
        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)

        # Reuse and parallelization factors
        layer.set_attr('strategy', 'resource')
        n_in, n_out = self.get_layer_mult_size(layer)
        self.set_target_reuse_factor(layer)
        self.set_closest_reuse_factor(layer, n_in, n_out)
        layer.set_attr('parallelization', layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1))

        # impl_filt_width & impl_filt_height determine the filter size post-Winograd transformation
        layer.set_attr('impl_filt_height', layer.get_attr('filt_height'))
        layer.set_attr('impl_filt_width', layer.get_attr('filt_width'))

        # Implementation:
        # - combination - at compile-time, the decision between Winograd and im2col is made
        # - im2col - specifically use im2col
        # - Winograd - use Winograd, if possible
        layer.set_attr('implementation', layer.model.config.get_layer_config_value(layer, 'Implementation', 'combination'))

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Not used yet as there is no codegen implementation of CNNs for Quartus backend

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        # We don't use RF yet
        if True:  # layer.model.config.is_resource_strategy(layer): ... Quartus only supports Dense resource multiplication
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')

        # Split weights for easier storage in on-chip memory and implementation in HLS
        weights_data = layer.weights['weight'].data
        rec_weights_data = layer.weights['recurrent_weight'].data
        bias_data = layer.weights['bias'].data

        weight_types = ['i', 'f', 'c', 'o']
        for i in range(0, 4):
            layer.add_weights_variable(
                name=f'weight_{weight_types[i]}',
                var_name=f'kernel_{weight_types[i]}_{{index}}',
                data=weights_data[
                    0 : layer.get_attr('n_in'), i * layer.get_attr('n_out') : (i + 1) * layer.get_attr('n_out')
                ],
                quantizer=layer.get_attr('weight_quantizer'),
                compression=None,
            )
            layer.add_weights_variable(
                name=f'recurrent_weight_{weight_types[i]}',
                var_name=f'recurrent_kernel_{weight_types[i]}_{{index}}',
                data=rec_weights_data[
                    0 : layer.get_attr('n_out'), i * layer.get_attr('n_out') : (i + 1) * layer.get_attr('n_out')
                ],
                quantizer=layer.get_attr('weight_quantizer'),
                compression=None,
            )
            layer.add_weights_variable(
                name=f'bias_{weight_types[i]}',
                var_name=f'bias_{weight_types[i]}_{{index}}',
                data=bias_data[i * layer.get_attr('n_out') : (i + 1) * (layer.get_attr('n_out'))],
                quantizer=layer.get_attr('weight_quantizer'),
                compression=None,
            )

    @layer_optimizer(SimpleRNN)
    def init_simple_rnn(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        # TODO - Consider setting and using RF
