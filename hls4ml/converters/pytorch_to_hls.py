import numpy as np
import torch

from hls4ml.model import ModelGraph


class PyTorchModelReader:
    """
    Pytorch data reader to be used for extracting relevant information during conversion.
    """

    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.input_shape = config['InputShape']

    def get_weights_data(self, layer_name, var_name):

        """Get weights data from layers.

        The hls layer classes are based on Keras's default parameters.
        Thus, this function will also need to account for some differences
        between Keras and Pytorch terminology.

        Parameters
        ----------
        layer_name : string
            layer's name in the Pytorch model
        var_name : string
            variable to be extracted

        Returns
        -------
        data : numpy array
            extracted weights data

        """

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
        if layer_name.split("_")[-1].isdigit() and len(layer_name.split("_")) > 1:
            layer_name = "_".join(layer_name.split("_")[:-1])

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

        data = {}  # this is just to shut up pre-commit, this function is broken somehow

        return data


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


# ----------------------------------------------------------------


def pytorch_to_hls(config):
    """Convert Pytorch model to hls model from configuration.

    Parameters
    ----------
    config: dict
        pytorch configuration from yaml file or passed through API.

    Returns
    -------
    ModelGraph : hls4ml model object.

    Notes
    -----
    Only sequential pytorch models are supported for now.
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

    # Map inputs of skipped and split (activation) layers
    # inputs_map = {}

    input_layers = None
    # output_layers = None

    # layer_config = None

    # Output shape tracking
    output_shapes = {}
    output_shape = None

    # Loop through layers
    print('Topology:')
    layer_counter = 0

    n_inputs = 0

    for node in traced_model.graph.nodes:

        # If part of a nn.Sequntial, the node name will start with an "_" which messes up the parsing
        if node.name[0] == "_":
            node.name = 'layer' + node.name

        if node.op == 'call_module':

            # modules that are part of a torch.nn.Sequential with name 'name' have target names 'name.x',
            # where x is an integer numbering the elements of the Sequential
            if "." in node.target:
                class_object = children[node.target.split(".")[0]][int(node.target.split(".")[1])]
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
            input_names = tuple([str(i) for i in node.args])
            input_shapes = [output_shapes[str(i)] for i in node.args]

            arguments = {}

            # for Softmax (and probably others)
            if hasattr(class_object, 'dim'):
                arguments['dim'] = class_object.dim
            # for Linear layer
            if hasattr(class_object, 'in_features'):
                arguments['in_features'] = class_object.in_features
            if hasattr(class_object, 'out_features'):
                arguments['out_features'] = class_object.out_features
            if hasattr(class_object, 'bias'):
                arguments['bias'] = class_object.bias
            # for Conv/Pool layers
            if hasattr(class_object, 'out_channels'):
                arguments['out_channels'] = class_object.out_channels
            if hasattr(class_object, 'kernel_size'):
                arguments['kernel_size'] = class_object.kernel_size
            if hasattr(class_object, 'stride'):
                arguments['stride'] = class_object.stride
            if hasattr(class_object, 'dilation'):
                arguments['dilation'] = class_object.dilation
            if hasattr(class_object, 'padding'):
                arguments['padding'] = class_object.padding
                if '1d' in pytorch_class and type(arguments['padding']) is tuple:
                    arguments['padding'] = arguments['padding'][0]
            # for BatchNorm layers
            if hasattr(class_object, 'eps'):
                arguments['eps'] = class_object.eps
            # for LeakyReLU, ELU, PreLU layers
            if hasattr(class_object, 'negative_slope'):
                arguments['alpha'] = class_object.negative_slope
            if hasattr(class_object, 'alpha'):
                arguments['alpha'] = class_object.alpha
            if pytorch_class == "PReLU":
                if hasattr(class_object, 'weight'):
                    arguments['alpha'] = class_object.weight.detach().numpy()[0]
            # for Threshold layers
            if hasattr(class_object, 'threshold'):
                arguments['threshold'] = class_object.threshold
            if hasattr(class_object, 'value'):
                arguments['value'] = class_object.value
            if pytorch_class == 'Threshold' and int(arguments['value']) != 0:
                raise Exception('values other than 0 for x < threshold not supported for Threshold layers')
            # for Pooling layers
            if 'Pool' in pytorch_class:
                if '2d' in pytorch_class and not type(arguments['kernel_size']) is tuple:
                    arguments['kernel_size'] = [arguments['kernel_size'], arguments['kernel_size']]
                elif '1d' in pytorch_class and type(arguments['kernel_size']) is tuple:
                    arguments['kernel_size'] = arguments['kernel_size'][0]
                if '2d' in pytorch_class and not type(arguments['padding']) is tuple:
                    arguments['padding'] = [arguments['padding'], arguments['padding']]
                elif '1d' in pytorch_class and type(arguments['padding']) is tuple:
                    arguments['padding'] = arguments['padding'][0]
                if '2d' in pytorch_class and not type(arguments['stride']) is tuple:
                    arguments['stride'] = [arguments['stride'], arguments['stride']]
                elif '1d' in pytorch_class and type(arguments['stride']) is tuple:
                    arguments['stride'] = arguments['stride'][0]

            # Process the layer
            print(input_shapes)
            layer, output_shape = layer_handlers[pytorch_class](
                pytorch_class, layer_name, input_names, input_shapes, arguments, reader, config
            )

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

            layer_counter += 1

        if node.op == 'placeholder':
            # 'placeholder' indicates an input layer. Multiple inputs are supported

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = input_shapes[n_inputs][1:]
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = input_shapes[n_inputs]
            n_inputs += 1

            layer_counter += 1

        if node.op == 'call_function':
            # Function calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            if node.name.split("_")[-1].isdigit():
                operation = "_".join(node.name.split("_")[:-1]).capitalize()
            else:
                operation = node.name.capitalize()

            # only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')

            layer_counter += 1

            # need a copy because kwargs are immutable
            arguments = {}
            for key in node.kwargs:
                arguments[key] = node.kwargs[key]
            layer_name = node.name

            # arguments of pooling layers need some massaging
            if 'pool' in operation:
                input_names = tuple([str(node.args[0])])
                arguments['kernel_size'] = int(node.args[1])
                if '2d' in operation and not type(arguments['kernel_size']) is tuple:
                    arguments['kernel_size'] = [arguments['kernel_size'], arguments['kernel_size']]
                if '2d' in operation and not type(arguments['padding']) is tuple:
                    arguments['padding'] = [arguments['padding'], arguments['padding']]
                if arguments['stride'] is None:
                    arguments['stride'] = arguments['kernel_size']
            elif 'Cat' in operation:
                input_names = tuple([str(i) for i in node.args[0]])
                arguments['axis'] = int(node.args[1])
            else:
                input_names = tuple([str(i) for i in node.args])

            input_shapes = [list(output_shapes[str(i)]) for i in list(input_names)]

            # Process the layer
            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, arguments, reader, config
            )

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

        if node.op == 'get_attr':
            # Deals with tensors that are member variables of the model class
            # We insert these tensors are input layer nodes into the hls4ML model graph
            if "." not in node.target:
                obj = getattr(model, node.name)
            else:
                obj = getattr(children[node.target.split('.')[0], node.name])

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = [None] + list(obj.size())
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = [None] + list(obj.size())
            n_inputs += 1

            layer_counter += 1

        if node.op == 'call_method':
            # Method calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            if node.name.split("_")[-1].isdigit():
                operation = "_".join(node.name.split("_")[:-1]).capitalize()
            else:
                operation = node.name.capitalize()

            # only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')

            layer_counter += 1

            # need a copy because kwargs are immutable
            arguments = {}
            for key in node.kwargs:
                arguments[key] = node.kwargs[key]
            layer_name = node.name

            if 'View' in operation:
                input_names = tuple([str(node.args[0])])
                arguments['target_shape'] = [int(i) for i in node.args[1:]]
                # View can have -1 as one as the dimensions,
                # leaving it to us to deduce it from the other dimensions and the overall size
                if -1 in arguments['target_shape']:
                    size = np.prod(input_shapes[0][1:])
                    for i in range(0, len(arguments['target_shape'])):
                        if arguments['target_shape'][i] == -1:
                            cl = arguments['target_shape'][:]
                            cl.remove(-1)
                            arguments['target_shape'][i] = int(size / np.prod(cl))

            else:
                input_names = tuple([str(i) for i in node.args])

            # Process the layer
            input_shapes = [list(output_shapes[str(i)]) for i in list(input_names)]

            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, arguments, reader, config
            )

            print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

    #################
    # Generate HLS
    #################

    print('Creating HLS model')
    hls_model = ModelGraph(config, reader, layer_list, inputs=input_layers)
    return hls_model
