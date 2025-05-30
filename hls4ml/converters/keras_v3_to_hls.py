import typing
from collections.abc import Callable, Sequence
from itertools import chain
from types import FunctionType
from typing import Any

import numpy as np

from hls4ml.model import ModelGraph

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor

from .keras_v3 import layer_handlers as v3_layer_handlers

T_kv3_handler = Callable[
    ['keras.Layer', Sequence['keras.KerasTensor'], Sequence['keras.KerasTensor']], tuple[dict[str, Any], ...]
]


def get_io_tensors(layer: 'keras.Layer', node_whitelist: set[int] | None = None):
    '''Given a keras layer, return a list of tuples of input and output
    tensors. If the layer is called only once (i.e., a layer is not used multiple times in the same model),
    the list will contain only one tuple.

    The layer must have been built before calling this function.

    Args:
        layer: The layer to get input and output tensors from.
        node_whitelist: If not None, only return tensors from nodes
            with ids in this set, used to filter out nodes that are not
            part of the model. Defaults to None.


    Returns:
        A list of tuples of input and output tensors. Each inner tuple
        contains two tuples: the first with input KerasTensors and the
        second with output KerasTensors.
    '''

    in_nodes = layer._inbound_nodes
    if node_whitelist is not None:
        in_nodes = [node for node in in_nodes if id(node) in node_whitelist]

    ret: list[tuple[tuple['KerasTensor', ...], tuple['KerasTensor', ...]]] = []
    for node in in_nodes:
        in_tensors = tuple(node.arguments.keras_tensors)
        out_tensors = tuple(node.outputs)
        ret.append((in_tensors, out_tensors))
    return ret


def resolve_dependency_relation(model: 'keras.Model'):
    '''Given a keras model, return the following information:
        - A list of input tensor names
        - A list of output tensor names
        - A list of (layer_name, input_tensor_names, output_tensor_names) tuples
        - A dictionary of tensor_name -> KerasTensor

    Args:
        model: The keras model to analyze.

    Returns:
        A tuple containing:
            - inp_tensor_names (tuple[str, ...]): A tuple of input tensor names.
            - out_tensor_names (tuple[str, ...]): A tuple of output tensor names.
            - layer_io (list[tuple[str, tuple[str, ...], tuple[str, ...]]]): A list of
                tuples, where each tuple contains the layer name, a tuple of its
                input tensor names, and a tuple of its output tensor names.
            - tensors (dict[str, KerasTensor]): A dictionary mapping tensor names
                to KerasTensor objects.
    '''

    tensors: dict[str, 'KerasTensor'] = {}
    'tensor_name -> KerasTensor'
    depends_on: dict[str, tuple[str, ...]] = {}
    'tensor_name -> {tensor_name}'
    layer_io: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    'layer_name -> ((input_tensor_names), (output_tensor_names))'

    inputs = tuple(t.name for t in model.inputs)
    outputs = tuple(t.name for t in model.outputs)
    node_whitelist = {id(node) for v in model._nodes_by_depth.values() for node in v}

    for layer in model.layers:
        for in_tensors, out_tensors in get_io_tensors(layer, node_whitelist):
            in_tensor_names = tuple(t.name for t in in_tensors)
            out_tensor_names = tuple(t.name for t in out_tensors)
            for t in chain(in_tensors, out_tensors):
                tensors[t.name] = t
            for o_name in out_tensor_names:
                depends_on[o_name] = in_tensor_names
            layer_io.append((layer.name, in_tensor_names, out_tensor_names))

    return inputs, outputs, layer_io, tensors


class UniqueName:
    '''Helper class to generate unique names for layers, if one being used multiple times.'''

    def __init__(self):
        self.used_names: set[str] = set('input')
        # input is reserved in hls4ml, avoid conflict with it

    def next_name(self, name: str):
        i = 0
        if name in self.used_names:
            while f'{name}_{i}' in self.used_names:
                i += 1
            name = f'{name}_{i}'
        self.used_names.add(name)
        return name

    def __call__(self, name: str):
        return self.next_name(name)

    def reset(self):
        self.used_names.clear()


class KerasV3HandlerDispatcher:
    '''Dispatcher class to handle different types of keras v3 layers.'''

    def __init__(self, layer_handlers: dict[str, T_kv3_handler], v2_layer_handlers=None):
        self.registry = layer_handlers
        self.v2_layer_handlers = v2_layer_handlers or {}

    def __call__(
        self, layer: 'keras.Layer', in_tensors: Sequence['keras.KerasTensor'], out_tensors: Sequence['keras.KerasTensor']
    ) -> tuple[dict[str, Any], ...]:
        assert layer.built, f'Layer {layer.name} is not built'

        ret = self.v3_call(layer, in_tensors, out_tensors)
        if ret is not None:
            return ret
        ret = self.v2_call(layer, in_tensors, out_tensors)
        if ret is not None:
            return ret

        raise ValueError(
            f'Layer {layer.__class__.__module__}.{layer.__class__.__name__} not found in either v3 or v2 handlers'
        )

    def v3_call(
        self, layer: 'keras.layers.Layer', inp_tensors: Sequence['KerasTensor'], out_tensors: Sequence['KerasTensor']
    ):
        cls_name = layer.__class__.__name__
        module = layer.__module__
        key = f'{module}.{cls_name}'

        # keras v3 handlers
        handler = self.registry.get(key, None)
        handler = handler or self.registry.get(cls_name, None)

        if handler is None:
            return None
        return handler(layer, inp_tensors, out_tensors)

    def v2_call(
        self, layer: 'keras.layers.Layer', inp_tensors: Sequence['KerasTensor'], out_tensors: Sequence['KerasTensor']
    ):
        # keras v2 handlers fallback
        print(f'v2 handler used for layer {layer.name}')

        import keras

        config = layer.get_config()
        layer_dict = {'config': config, 'class_name': layer.__class__.__name__}

        class IsolatedLayerReader:
            def get_weights_data(self, layer_name, var_name):
                assert layer_name == layer.name, f'Processing {layer.name}, but handler tried to read {layer_name}'
                for w in layer.weights:
                    if var_name in w.name:
                        return np.array(w)
                return None

        reader = IsolatedLayerReader()
        input_shapes = [list(t.shape) for t in inp_tensors]
        input_names = [t.name for t in inp_tensors]
        output_names = [t.name for t in out_tensors]
        key = layer.__class__.__name__
        handler = self.v2_layer_handlers.get(key, None)
        if handler is None:
            return None

        ret, _ = handler(layer_dict, input_names, input_shapes, reader)
        ret['output_keras_tensor_names'] = output_names
        ret['input_keras_tensor_names'] = input_names
        ret = (ret,)

        activation = getattr(layer, 'activation', None)
        if activation not in (keras.activations.linear, None):
            assert isinstance(activation, FunctionType), f'Activation function for layer {layer.name} is not a function'
            intermediate_tensor_name = f'{output_names[0]}_activation'
            ret[0]['output_keras_tensor_names'] = (intermediate_tensor_name,)
            act_cls_name = activation.__name__
            act_config = {
                'class_name': 'Activation',
                'activation': act_cls_name,
                'name': f'{layer.name}_{act_cls_name}',
                'input_keras_tensor_names': (intermediate_tensor_name,),
                'output_keras_tensor_names': output_names,
            }
            ret = *ret, act_config
        return ret


def parse_keras_v3_model(model: 'keras.Model'):
    '''Parse a keras model into a list of dictionaries, each
    representing a layer in the HLS model, and a list of input and
    output layer names.

    Args:
        model: keras.Model

    Returns:
        A tuple containing:
            - layer_list (list[dict[str, Any]]): A list of dictionaries,
                each representing a layer in the HLS model.
            - input_layer_names (list[str]): A list of input layer names.
            - output_layer_names (list[str]): A list of output layer names.
            - batch_output_shapes (list[list[int]]): A list of output shapes.

    Raises:
        ValueError: If a circular dependency is detected.
    '''

    assert model.built, 'Model must be built before parsing'

    import keras

    if isinstance(model, keras.Sequential):
        model = model._functional  # everything is functional under the hood lol

    from .keras_v2_to_hls import layer_handlers as v2_layer_handlers  # Delayed import to avoid circular import

    keras_v3_dispatcher = KerasV3HandlerDispatcher(v3_layer_handlers, v2_layer_handlers)

    model_inputs, model_outputs, dependency, tensors = resolve_dependency_relation(model)

    satisfied = set()

    unique_name = UniqueName()

    layer_list: list[dict[str, Any]] = []

    while any(t not in satisfied for t in model_outputs):
        # Until all tensors in the model are satisfied
        for i, (layer_name, in_tensor_names, out_tensor_names) in enumerate(dependency):
            if not all(t in satisfied for t in in_tensor_names):
                continue  # Skip layer if some inputs are not ready
            if all(t in satisfied for t in out_tensor_names):
                continue  # Skip layer if the outputs are already satisfied

            layer: 'keras.Layer' = model.get_layer(layer_name)
            inp_tensors = [tensors[t] for t in in_tensor_names]
            out_tensors = [tensors[t] for t in out_tensor_names]

            _configs = keras_v3_dispatcher(layer, inp_tensors, out_tensors)
            # Dispatch to v3 handler if available, else fallback to v2 handler

            # Prevent name conflicts. If a layer is used multiple times, add a suffix to the name.
            # At this stage connections between modules are recorded by i/o keras tensor names
            for _conf in _configs:
                _conf['name'] = unique_name(_conf['name'])

            layer_list.extend(_configs)  # Add the layer to the list
            satisfied.update(out_tensor_names)  # Mark the outputs as satisfied
            dependency.pop(i)
            break  # Restart the loop to add another layer
        else:
            # If no layer was added in the loop, then there is a circular dependency
            raise ValueError('Circular dependency detected')

    # Mark inputs[inp layer name] for ModelGraph to parse from i/o keras tensor names
    provides: dict[str, str] = {}  # tensor_name -> src_layer_name
    for conf in layer_list:
        for out_name in conf['output_keras_tensor_names']:
            provides[out_name] = conf['name']
        inputs = [provides[tname] for tname in conf['input_keras_tensor_names']]
        conf['inputs'] = inputs

    input_layer_names = [provides[tname] for tname in model_inputs]
    output_layer_names = [provides[tname] for tname in model_outputs]
    batch_output_shapes = [list(tensors[tname].shape) for tname in model_outputs]

    return layer_list, input_layer_names, output_layer_names, batch_output_shapes


def keras_v3_to_hls(config):
    layer_list, input_layers, output_layers, _ = parse_keras_v3_model(config['KerasModel'])
    return ModelGraph.from_layer_list(config, layer_list, input_layers, output_layers)
