import math

import numpy as np

from hls4ml.model import ModelGraph
from hls4ml.model.quantizers import BrevitasQuantizer
from hls4ml.model.types import FixedPrecisionType
from hls4ml.utils.dependency import requires


class PyTorchModelReader:
    """
    PyTorch reader to extract weights data.
    """

    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.input_shape = config['InputShape']

    def get_weights_data(self, layer_name, var_name):
        data = None

        tensorName = layer_name + '.' + var_name

        if tensorName in self.state_dict:
            data = self.state_dict[tensorName].numpy()

        return data


class PyTorchFileReader(PyTorchModelReader):  # Inherit get_weights_data method
    def __init__(self, config):
        import torch

        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])

        # Get input tensor's shape
        self.input_shape = config.get('InputShape')

        if self.input_shape is None:
            raise Exception('Must specify input shape ("InputShape") in config!')

        # Convert it to a list
        self.input_shape = self.input_shape.strip('(,)').split(',')
        self.input_shape = [None if n == 'None' else int(n) for n in self.input_shape]

        self.state_dict = self.torch_model.state_dict()


def get_weights_data(data_reader, layer_name, var_name):
    if not isinstance(var_name, (list, tuple)):
        var_name = [var_name]

    data = [data_reader.get_weights_data(layer_name, var) for var in var_name]

    if len(data) == 1:
        return data[0]
    else:
        return (*data,)


def convert_uaq_to_apfixed(bitwidth, scale_factor):
    """
    parameters:
    bitwidth: int
    scale_factor: float
    zero_point: float

    return:
    int_bitwidth: int
    fract_bitwidth: int
    """
    fract_bitwidth = -math.log2(scale_factor)
    int_bitwidth = bitwidth - fract_bitwidth

    return (fract_bitwidth, int_bitwidth)


def _quant_tensor_precision(quant_tensor):
    """Number of integer and fractional bits of the ap_fixed type that exactly holds a brevitas tensor.

    The integer count includes the sign bit, matching hls4ml's FixedPrecisionType.
    """
    width = int(quant_tensor.bit_width)
    scale = float(quant_tensor.scale.detach())
    mantissa, _ = np.frexp(scale)
    if mantissa != 0.5:
        raise Exception(
            """Non-power of 2 quantization of weights not supported when injecting brevitas models.
            Please used QONNX instead."""
        )
    fract_bitwidth, int_bitwidth = convert_uaq_to_apfixed(width, scale)
    return int(int_bitwidth), int(fract_bitwidth), bool(quant_tensor.signed)


def brevitas_quant_to_hls(quant_tensors, transpose=False):
    """Turn one or more brevitas quant tensors into a weight array plus the matching hls4ml quantizer.

    Passing a sequence concatenates the tensors along axis 0, which is how the per-gate LSTM weights combine
    into the single tensor the HLS code expects. Each tensor carries its own quantizer, so the returned type
    is widened to whatever exactly represents all of them; it collapses to the common type when the scales
    agree. Transposing afterwards matches the keras order used in the HLS code, as in parse_rnn_layer.
    """
    # brevitas' quant tensors are NamedTuples, so an isinstance check against tuple would happily iterate
    # over their fields instead of treating a single tensor as one item
    if hasattr(quant_tensors, 'bit_width'):
        quant_tensors = [quant_tensors]

    precisions = [_quant_tensor_precision(t) for t in quant_tensors]
    integer = max(p[0] for p in precisions)
    fractional = max(p[1] for p in precisions)
    signed = any(p[2] for p in precisions)
    width = integer + fractional

    data = np.concatenate([t.detach().value.numpy() for t in quant_tensors], axis=0)
    if transpose:
        data = data.transpose()

    quantizer = BrevitasQuantizer(width, FixedPrecisionType(width=width, integer=integer, signed=signed))
    return data, quantizer


# embed quantization information into the layer dictionary for a Quant layer
# so that this layer can be added to the model
def addQuantizationParameters(layer, quant_object, quant_type, act=False):
    if not act:
        # currently not used, might be use later for non-power-of-2 scales
        bit_width = int(quant_object.bit_width)
        signed = quant_object.signed
        scale = float(quant_object.scale)
        zeropoint = float(quant_object.zero_point)
        if signed:
            narrow = True
        else:
            narrow = False
        rounding_mode = 'ROUND'
    else:
        bit_width = int(quant_object.bit_width())
        signed = quant_object.is_signed
        scale = float(quant_object.scale())
        zeropoint = float(quant_object.zero_point())
        narrow = quant_object.is_narrow_range
        # trunc quantizers report their rounding mode in lower case, unlike the activation quantizers
        rounding_mode = quant_object.rounding_mode.upper()
        if signed is None:
            # A trunc quantizer inherits signedness from whatever it is fed, so it cannot report it here.
            # Assume signed, which represents unsigned data correctly at the cost of one bit of range.
            signed = True

    layer[f'{quant_type}_quantization'] = {
        'bit_width': bit_width,
        'signed': signed,
        'scale': scale,
        'zeropoint': zeropoint,
        'narrow': narrow,
        'rounding_mode': rounding_mode,
    }
    return layer


# ----------------------Layer handling--------------------- #
layer_handlers = {}


def register_pytorch_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception(f'Layer {layer_name} already registered')
    else:
        layer_handlers[layer_name] = handler_func


def get_supported_pytorch_layers():
    return list(layer_handlers.keys())


def pytorch_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


# map names of operations between torch.nn and torch.nn.functionals
layer_name_map = {
    'relu': 'ReLU',
    'tanh': 'Tanh',
    'leaky_relu': 'LeakyReLU',
    'elu': 'ELU',
    'prelu': 'PReLU',
    'sigmoid': 'Sigmoid',
    '_threshold': 'Threshold',
    'softmax': 'Softmax',
    'max_pool1d': 'MaxPool1d',
    'max_pool2d': 'MaxPool2d',
    'avg_pool1d': 'AvgPool1d',
    'avg_pool2d': 'AvgPool2d',
    'flatten': 'Flatten',
    'view': 'View',
}


# ----------------------------------------------------------------


def parse_pytorch_model(config, verbose=True):
    """Convert PyTorch model to hls4ml ModelGraph.

    Args:
        config (dict): The conversion config

    Raises:
        Exception: On unsupported features of the model.

    Returns:
        ModelGraph: hls4ml model object.
    """
    import torch

    from hls4ml.utils.torch import CustomFXTracer

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if verbose:
        print('Interpreting Model ...')
    reader = PyTorchFileReader(config) if isinstance(config['PytorchModel'], str) else PyTorchModelReader(config)
    if type(reader.input_shape) is tuple:
        input_shapes = [list(reader.input_shape)]
    else:
        input_shapes = list(reader.input_shape)
    # first element needs to 'None' as placeholder for the batch size, insert it if not present
    input_shapes = [[None] + list(shape) if shape[0] is not None else list(shape) for shape in input_shapes]

    model = reader.torch_model

    # dict of layer objects in non-traced form for access later on
    children = {c[0]: c[1] for c in model.named_children()}
    # use symbolic_trace to get a full graph of the model

    tracer = CustomFXTracer()
    traced_model = tracer.trace(model)
    # Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'QuantDropout', 'Sequential']

    # Define layers with associated Quantizer
    quantizer_layers = [
        'PQDense',
        'PQBatchNorm1d',
        'PQBatchNorm2d',
        'PQConv1d',
        'PQConv2d',
        'PQAvgPool1d',
        'PQAvgPool2d',
        'PQActivation',
    ]

    # All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers

    # Map inputs of skipped and split (activation) layers
    inputs_map = {}

    input_layers = []
    output_layers = []

    # Output shape tracking
    output_shapes = {}
    output_shape = None

    # Loop through layers
    if verbose:
        print('Topology:')
    layer_counter = 0

    n_inputs = 0

    # check for constant nodes
    merge_layers = ['add', 'mul', 'sub', 'fmin', 'fmax']
    i = 0  # count number of consts and use it in the name
    for node in traced_model.nodes:
        if node.name.split('_')[0] in merge_layers:
            for arg in node.args:
                if np.isscalar(arg):
                    # add an input node with the constant value
                    new_node = traced_model.placeholder(name='const_' + str(i), type_expr=torch.Tensor, default_value=arg)
                    node.prepend(new_node)
                    node.update_arg(1, new_node)
                    i += 1

    traced_model.lint()

    for node in traced_model.nodes:
        if node.op == 'call_module':
            # modules that are part of a torch.nn.Sequential with name 'name' have target names 'name.x',
            # where x is an integer numbering the elements of the Sequential
            if '.' in node.target:
                fqn_path = node.target.split('.')
                sub_children = dict(children[fqn_path[0]].named_children())
                for name in fqn_path[1:-1]:
                    sub_children = dict(sub_children[name].named_children())
                sub_children[fqn_path[-1]]
                class_object = sub_children[fqn_path[-1]]
            else:
                class_object = children[node.target]

            pytorch_class = class_object.__class__.__name__

            if pytorch_class not in supported_layers:
                raise Exception(f'Unsupported layer {pytorch_class}')

            if 'IOType' in config.keys():
                if 'QuantUpsampl' in pytorch_class and config['IOType'] == 'io_stream':
                    raise Exception('Quant upsampling layers currently not supported with io_stream')

            if layer_counter != 0:
                input_shapes = [output_shape]  # In case there are multiple inputs

            layer_name = node.name

            # Handle skipped layers
            if pytorch_class in skip_layers:
                if pytorch_class == 'Sequential':  # Ignore the mother module's class name
                    continue

                # Assuming only one input
                parent_input = [str(i) for i in node.args][0]
                inputs_map[layer_name] = inputs_map.get(parent_input, parent_input)

                output_shapes[layer_name] = input_shapes[0]

                continue

            # Increment the layer counter after initial screenings
            if pytorch_class in supported_layers:
                layer_counter += 1

            # parse info from class object
            input_names = [inputs_map.get(str(i), str(i)) for i in node.args]
            if pytorch_class in ['RNN', 'GRU', 'LSTM', 'QuantRNN', 'QuantLSTM']:
                input_shapes = []
                input_names = []
                for arg in node.args:
                    if isinstance(arg, tuple):
                        for input in arg:
                            input_shapes.append(output_shapes[str(input)])
                            input_names.append(inputs_map.get(str(input), str(input)))
                    else:
                        input_shapes.append(output_shapes[str(arg)])
                        input_names.append(inputs_map.get(str(arg), str(arg)))

            # if a 'getitem' is the input to a node, step back in the graph to find the real source of the input
            elif 'getitem' in node.args[0].name:

                def resolve_getitem_source(node_name, visited=None):
                    """Recursively resolve nested getitem calls to find the actual source node."""
                    if visited is None:
                        visited = set()

                    if node_name in visited:
                        raise Exception(f'Circular reference detected in getitem chain: {node_name}')
                    visited.add(node_name)

                    for tmp_node in traced_model.nodes:
                        if tmp_node.name == node_name:
                            if 'getitem' in tmp_node.args[0].name:
                                return resolve_getitem_source(tmp_node.args[0].name, visited)
                            else:
                                return tmp_node.args[0]
                    raise Exception(f'Could not find source node for getitem: {node_name}')

                source_node = resolve_getitem_source(node.args[0].name)
                input_names = [inputs_map.get(str(source_node), str(source_node))]
                input_shapes = [output_shapes[str(source_node)]]
                node.args = [source_node]
            else:
                input_shapes = [output_shapes[str(i)] for i in node.args]
            # for Conv layers
            if 'Conv' in pytorch_class:
                if not class_object.padding_mode == 'zeros':
                    raise Exception('Padding modes other than "zeros" not implemented yet')
                if not class_object.groups == 1:
                    raise Exception('Non-default options for groups not implemented yet')

            # Process the layer
            layer, output_shape = layer_handlers[pytorch_class](
                pytorch_class, layer_name, input_names, input_shapes, node, class_object, reader, config
            )

            if isinstance(layer, dict):
                if verbose:
                    print(
                        'Layer name: {}, layer type: {}, input shape: {}'.format(
                            layer['name'],
                            layer['class_name'],
                            input_shapes,
                        )
                    )
                layer_list.append(layer)

                assert output_shape is not None
                output_shapes[layer['name']] = output_shape

                layer_counter += 1

            else:
                for lay, out_shape in zip(layer, output_shape):
                    if verbose:
                        print(
                            'Layer name: {}, layer type: {}, input shape: {}'.format(
                                lay['name'],
                                lay['class_name'],
                                input_shapes,
                            )
                        )
                    layer_list.append(lay)

                    assert out_shape is not None
                    output_shapes[lay['name']] = out_shape

                    layer_counter += 1

                # Handle layers with output quantizer (assuming only one output)
                if pytorch_class in quantizer_layers:
                    if getattr(class_object, 'quantize_output', False) and hasattr(class_object, 'output_quantizer'):
                        inputs_map[layer_name] = layer[-1]['name']

        if node.op == 'placeholder':
            # 'placeholder' indicates an input layer. Multiple inputs are supported

            input_layer = {}
            input_layer['name'] = node.name

            if 'const' in node.name:
                pytorch_class = 'Constant'
                layer, output_shape = layer_handlers[pytorch_class](pytorch_class, node.name, node)

                layer_list.append(layer)

                assert output_shape is not None
                output_shapes[layer['name']] = output_shape

            else:
                input_layer['class_name'] = 'InputLayer'
                input_layer['input_shape'] = list(input_shapes[n_inputs][1:])
                layer_list.insert(n_inputs, input_layer)

                output_shapes[input_layer['name']] = list(input_shapes[n_inputs])

                input_layers.append(input_layer['name'])
                n_inputs += 1

            layer_counter += 1

        if node.op == 'call_function':
            # Function calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            operation = node.name
            if node.name.split('_')[-1].isdigit():
                operation = '_'.join(node.name.split('_')[:-1])

            if operation in layer_name_map:
                operation = layer_name_map[operation]

            # only a limited number of functions are supported
            if operation == 'getitem':
                continue
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')
            if operation == 'PReLU' or operation == 'batch_norm' or operation == 'conv1d' or operation == 'conv2d':
                raise Exception(
                    f'Function {operation} cannot be parsed as torch.nn.functional. Use the torch.nn implementation instead'
                )

            layer_name = node.name

            layer_counter += 1

            input_names = [inputs_map.get(str(i), str(i)) for i in node.all_input_nodes]
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            # Process the layer
            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            if verbose:
                print(
                    'Layer name: {}, layer type: {}, input shape: {}'.format(
                        layer['name'], layer['class_name'], input_shapes
                    )
                )
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

        if node.op == 'get_attr':
            # Deals with tensors that are member variables of the model class
            # We insert these tensors are input layer nodes into the hls4ML model graph
            if '.' not in node.target:
                obj = getattr(model, node.name)
            else:
                obj = getattr(children[node.target.split('.')[0], node.name])

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = [None] + list(obj.size())
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = [None] + list(obj.size())
            input_layers.append(input_layer['name'])
            n_inputs += 1

            layer_counter += 1

        if node.op == 'call_method':
            # Method calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            operation = node.name
            if node.name.split('_')[-1].isdigit():
                operation = '_'.join(node.name.split('_')[:-1])

            if operation in layer_name_map:
                operation = layer_name_map[operation]

            # only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')

            layer_name = node.name

            layer_counter += 1

            input_names = [inputs_map.get(str(i), str(i)) for i in node.all_input_nodes]

            # Process the layer
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            if verbose:
                print(
                    'Layer name: {}, layer type: {}, input shape: {}'.format(
                        layer['name'], layer['class_name'], input_shapes
                    )
                )
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

    if len(input_layers) == 0:
        input_layers = None

    for layer in layer_list:
        if layer['class_name'] == 'InputLayer':
            continue
        is_input = False
        for lay in layer_list:
            if 'inputs' not in lay.keys():
                continue
            if layer['name'] in lay['inputs']:
                is_input = True
        if not is_input:
            output_layers.append(layer['name'])
    return layer_list, input_layers, output_layers


@requires('_torch')
def pytorch_to_hls(config):
    layer_list, input_layers, output_layers = parse_pytorch_model(config)
    return ModelGraph.from_layer_list(config, layer_list, inputs=input_layers, outputs=output_layers)
