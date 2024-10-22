import numpy as np

import hls4ml
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.config import SUPPORTED_LAYERS


class hls4mlAttributes:
    '''
    A class for storing hls4ml information of a single layer

    Args:
        n_in (int): Number of inputs (rows) for Dense matrix multiplication
        n_out (int): Number of outputs (cols) for Dense matrix multiplication
        io_type (string): io_parallel or io_stream
        strategy (string): Resource or Latency
        weight_precision (FixedPrecisionType): Layer weight precision
        output_precision (FixedPrecisionType): Layer output precision
        reuse_factor (int): Layer reuse factor
        parallelization_factor (int): Layer parallelization factor - [applicable to io_parallel Conv2D]
    '''

    def __init__(
        self, n_in, n_out, io_type, strategy, weight_precision, output_precision, reuse_factor, parallelization_factor=1
    ):
        if not isinstance(weight_precision, (FixedPrecisionType, IntegerPrecisionType)):
            raise Exception('Layer weight precision is not in valid format')

        if not isinstance(output_precision, (FixedPrecisionType, IntegerPrecisionType)):
            raise Exception('Layer weight precision is not in valid format')

        if strategy not in ('Latency', 'latency', 'Resource', 'resource'):
            raise Exception('Unknown layer strategy')

        if io_type not in ('io_parallel', 'io_stream'):
            raise Exception('Unknown IO type')

        self.n_in = n_in
        self.n_out = n_out
        self.io_type = io_type
        self.strategy = strategy
        self.weight_precision = weight_precision
        self.output_precision = output_precision
        self.reuse_factor = reuse_factor
        self.parallelization_factor = parallelization_factor


class OptimizationAttributes:
    '''
    A class for storing layer optimization attributes

    Args:
        structure_type (enum): Targeted structure - unstructured, structured, pattern, block
        pruning (boolean): Should pruning be applied to the layer
        weight_sharing (boolean): Should weight sharing be applied to the layer
        block_shape (tuple): Block shape if structure_type == block
        pattern_offset (int): Length of each pattern if structure_type == pattern
        consecutive_patterns (int): How many consecutive patterns are grouped together if structure_type == pattern

    Notes:
        - In the case of hls4ml, pattern_offset is equivalent to the number of weights processed in parallel
        - The pattern_offset is n_in * n_out / reuse_factor; default case (=1) is equivalent to no unrolling
    '''

    def __init__(
        self,
        structure_type=SUPPORTED_STRUCTURES.UNSTRUCTURED,
        pruning=False,
        weight_sharing=False,
        block_shape=(1, 1),
        pattern_offset=1,
        consecutive_patterns=1,
    ):
        if not isinstance(structure_type, SUPPORTED_STRUCTURES):
            raise Exception(f'{self.__class__.__name__} unknown structure type')

        self.structure_type = structure_type
        self.pruning = pruning
        self.weight_sharing = weight_sharing
        self.block_shape = block_shape
        self.pattern_offset = pattern_offset
        self.consecutive_patterns = consecutive_patterns


class LayerAttributes:
    '''
    A class for storing layer information

    Args:
        name (string): Layer name
        layer_type (keras.Layer): Layer type (e.g. Dense, Conv2D etc.)
        inbound_layers (list): List of parent nodes, identified by name
        weight_shape (tuple): Layer weight shape
        input_shape (tuple): Layer input shape
        output_shape (tuple): Layer output shape
        optimizable (bool): Should optimizations (pruning, weight sharing) be applied to this layer
        optimization_attributes (OptimizationAttributes): Type of optimization,
            pruning or weight sharing, block shape and pattern offset
        args (dict): Additional information,
            e.g. hls4mlAttributes; dictionary so it can be generic enough for different platforms
    '''

    def __init__(
        self,
        name,
        layer_type,
        inbound_layers,
        weight_shape,
        input_shape,
        output_shape,
        optimizable,
        optimization_attributes,
        args,
    ):
        self.name = name
        self.layer_type = layer_type
        self.inbound_layers = inbound_layers
        self.weight_shape = weight_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizable = optimizable
        self.optimization_attributes = optimization_attributes
        self.args = args

    def update_args(self, updates):
        self.args.update(updates)

    def __str__(self):
        return (
            f'name: {self.name}, '
            f'layer_type: {self.layer_type}, '
            f'inbound_layers: {self.inbound_layers}, '
            f'weight_shape: {self.weight_shape}, '
            f'input_shape: {self.input_shape}, '
            f'output_shape: {self.output_shape}, '
            f'optimizable: {self.optimizable}, '
            f'optimization_attributes: {self.optimization_attributes}, '
            f'args: {self.args}, '
        )


def get_attributes_from_keras_model(model):
    '''
    Given a Keras model, builds a dictionary of class attributes
    Additional arguments (e.g. reuse factor), depend on the target hardware platform and are inserted later
    Per-layer pruning sype (structured, pattern etc.), depend on the pruning objective and are inserted later

    Args:
        model (keras.model): Model to extract attributes from

    Returns:
        model_attributes (dict): Each key corresponds to a layer name, values are instances of LayerAttribute
    '''
    is_sequential = model.__class__.__name__ == 'Sequential'
    model_attributes = {}

    for i, layer in enumerate(model.layers):
        inbound_layers = []
        if is_sequential and i > 0:
            inbound_layers.append(model.layers[i - 1])
        elif not is_sequential:
            nodes = model.get_config()['layers'][i]['inbound_nodes']
            if len(nodes) > 0:
                inbound_layers.append(node[0] for node in nodes[0])

        layer_weights = layer.get_weights()
        weight_shape = layer_weights[0].shape if len(layer_weights) > 0 else ()

        model_attributes[layer.name] = LayerAttributes(
            layer.name,
            layer.__class__,
            inbound_layers,
            weight_shape,
            layer.input_shape[1:],
            layer.output_shape[1:],
            False,
            OptimizationAttributes(),
            {},
        )

    return model_attributes


def get_attributes_from_keras_model_and_hls4ml_config(model, config):
    '''
    Given a Keras model and hls4ml configuration, builds a dictionary of class attributes
    Per-layer pruning sype (structured, pruning etc.), depend on the pruning objective and are inserted later

    Args:
        model (keras.model): Model to extract attributes from
        config (dict): hls4ml dictionary

    Returns:
        model_attributes (dict): Each key corresponds to a layer name, values are LayerAttribute instances
    '''

    # Extract Keras attributes
    model_attributes = get_attributes_from_keras_model(model)

    # Extract hls4ml attributes
    io_type = config['IOType']
    default_reuse_factor = config['Model']['ReuseFactor']
    default_strategy = config['Model']['Strategy']
    default_precision = config['Model']['Precision']

    # Build dictionary
    for layer in model_attributes:
        if model_attributes[layer].layer_type in SUPPORTED_LAYERS:
            n_in, n_out = __get_layer_mult_size(model_attributes[layer])
            layer_config = config['LayerName'][layer] if layer in config['LayerName'] else {}
            reuse_factor = layer_config['ReuseFactor'] if 'ReuseFactor' in layer_config else default_reuse_factor
            parallelization_factor = layer_config['ParallelizationFactor'] if 'ParallelizationFactor' in layer_config else 1
            strategy = layer_config['Strategy'] if 'Strategy' in layer_config else default_strategy
            weight_precision = (
                layer_config['Precision']['weight'] if 'weight' in layer_config['Precision'] else default_precision
            )
            weight_precision = hls4ml.backends.fpga.fpga_backend.FPGABackend.convert_precision_string(weight_precision)
            output_precision = (
                layer_config['Precision']['result'] if 'result' in layer_config['Precision'] else default_precision
            )
            output_precision = hls4ml.backends.fpga.fpga_backend.FPGABackend.convert_precision_string(output_precision)

            hls4ml_attributes = hls4mlAttributes(
                n_in, n_out, io_type, strategy, weight_precision, output_precision, reuse_factor, parallelization_factor
            )
            model_attributes[layer].update_args({'hls4ml_attributes': hls4ml_attributes})

    return model_attributes


def __get_layer_mult_size(attributes):
    '''
    Helper function to calculate layer multiplication size
    '''
    if 'Dense' in attributes.layer_type.__name__:
        n_in = np.prod(attributes.input_shape)
        n_out = np.prod(attributes.output_shape)
        return n_in, n_out

    if 'Conv' in attributes.layer_type.__name__:
        n_in = np.prod(attributes.weight_shape[0:-2])
        n_out = attributes.weight_shape[-1]
        return n_in, n_out

    raise Exception(f'Cannot get mult size for layer {attributes.name}')
