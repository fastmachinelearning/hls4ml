import numpy as np
import torch

from hls4ml.model import ModelGraph


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

        # Parameter mapping from pytorch to keras
        torch_paramap = {
            # Conv
            'kernel': 'weight',
            # Batchnorm
            'gamma': 'weight',
            # Activiation
            'alpha': 'weight',
            'beta': 'bias',
            'moving_mean': 'running_mean',
            'moving_variance': 'running_var',
        }

        # Workaround for naming schme in nn.Sequential,
        # have to remove the prefix we previously had to add to make sure the tensors are found
        if 'layer_' in layer_name:
            layer_name = layer_name.split('layer_')[-1]

        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')

        elif var_name in list(torch_paramap.keys()):
            var_name = torch_paramap[var_name]

        # if a layer is reused in the model, torch.FX will append a "_n" for the n-th use
        # have to snap that off to find the tensors
        if layer_name.split('_')[-1].isdigit() and len(layer_name.split('_')) > 1:
            layer_name = '_'.join(layer_name.split('_')[:-1])

        if layer_name + '.' + var_name in self.state_dict:
            data = self.state_dict[layer_name + '.' + var_name].numpy()
            return data

        else:
            return None


class PyTorchFileReader(PyTorchModelReader):  # Inherit get_weights_data method
    def __init__(self, config):
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


# map names of operations between toch.nn and torch.nn.functionals
layer_name_map = {
    'relu': 'ReLU',
    'leaky_relu': 'LeakyReLU',
    'elu': 'ELU',
    'prelu': 'PReLU',
    'sigmoid': 'Sigmoid',
    'layer_threshold': 'Threshold',
    'softmax': 'Softmax',
    'max_pool1d': 'MaxPool1d',
    'max_pool2d': 'MaxPool2d',
    'avg_pool1d': 'AvgPool1d',
    'avg_pool2d': 'AvgPool2d',
}


# ----------------------------------------------------------------


def pytorch_to_hls(config):
    """Convert PyTorch model to hls4ml ModelGraph.

    Args:
        config (dict): The conversion config

    Raises:
        Exception: On unsupported features of the model.

    Returns:
        ModelGraph: hls4ml model object.
    """

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    print('Interpreting Model ...')

    reader = PyTorchFileReader(config) if isinstance(config['PytorchModel'], str) else PyTorchModelReader(config)
    if type(reader.input_shape) is tuple:
        input_shapes = [list(reader.input_shape)]
    else:
        input_shapes = list(reader.input_shape)

    model = reader.torch_model

    # dict of layer objects in non-traced form for access lateron
    children = {c[0]: c[1] for c in model.named_children()}
    # use symbolic_trace to get a full graph of the model
    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    # Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten', 'Sequential']

    # All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers

    input_layers = []

    # Output shape tracking
    output_shapes = {}
    output_shape = None

    # Loop through layers
    print('Topology:')
    layer_counter = 0

    n_inputs = 0

    for node in traced_model.graph.nodes:
        # If part of a nn.Sequntial, the node name will start with an "_" which messes up the parsing
        if node.name[0] == '_':
            node.name = 'layer' + node.name

        if node.op == 'call_module':
            # modules that are part of a torch.nn.Sequential with name 'name' have target names 'name.x',
            # where x is an integer numbering the elements of the Sequential
            if '.' in node.target:
                class_object = children[node.target.split('.')[0]][int(node.target.split('.')[1])]
            else:
                class_object = children[node.target]

            pytorch_class = class_object.__class__.__name__

            if pytorch_class not in supported_layers:
                raise Exception(f'Unsupported layer {pytorch_class}')

            if layer_counter != 0:
                input_shapes = [output_shape]  # In case there are multiple inputs

            layer_name = node.name

            # Handle skipped layers
            if pytorch_class in skip_layers:
                if pytorch_class == 'Sequential':  # Ignore the mother module's class name
                    continue

                if pytorch_class == 'Flatten':
                    output_shapes[layer_name] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
                else:
                    output_shapes[layer_name] = input_shapes[0]
                continue

            # Increment the layer counter after initial screenings
            if pytorch_class in supported_layers:
                layer_counter += 1

            # parse info from class object
            input_names = [str(i) for i in node.args]
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

        if node.op == 'placeholder':
            # 'placeholder' indicates an input layer. Multiple inputs are supported

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = list(input_shapes[n_inputs][1:])
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = input_shapes[n_inputs]
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
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')
            if operation == 'PReLU' or operation == 'batch_norm' or operation == 'conv1d' or operation == 'conv2d':
                raise Exception(
                    f'Function {operation} cannot be parsed as torch.nn.functional. Use the torch.nn implementation instead'
                )

            layer_name = node.name

            layer_counter += 1

            input_names = [str(i) for i in node.all_input_nodes]
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            # Process the layer
            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
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

            if 'View' in operation:
                input_names = [str(node.args[0])]
            else:
                input_names = [str(i) for i in node.args]

            # Process the layer
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

    if len(input_layers) == 0:
        input_layers = None

    print('Creating HLS model')
    hls_model = ModelGraph(config, layer_list, inputs=input_layers)
    return hls_model
