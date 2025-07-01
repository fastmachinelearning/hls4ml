import concurrent.futures
import copy
import ctypes
import importlib.util
import os
import platform
import shutil
import threading
import uuid
from collections import OrderedDict

import numpy as np
import numpy.ctypeslib as npc

from hls4ml.backends import get_backend
from hls4ml.model.flow import get_flow
from hls4ml.model.layers import layer_map
from hls4ml.model.optimizer import get_available_passes, optimize_model
from hls4ml.model.types import Serializable
from hls4ml.utils.string_utils import convert_to_snake_case


class HLSConfig(Serializable):
    """The configuration class as stored in the ModelGraph.

    Args:
        config (dict):  The configuration dictionary
    """

    def __init__(self, config):
        self.config = config
        self.backend = get_backend(self.config.get('Backend', 'Vivado'))

        self.model_precision = {}
        self.layer_type_precision = {}
        self.layer_name_precision = {}

        self.model_rf = None
        self.layer_type_rf = {}
        self.layer_name_rf = {}

        self.model_targ_cycles = None
        self.layer_type_targ_cycles = {}
        self.layer_name_targ_cycles = {}

        self.model_strategy = convert_to_snake_case('Latency')
        self.layer_type_strategy = {}
        self.layer_name_strategy = {}

        self.model_conv_implementation = 'LineBuffer'
        self.layer_type_conv_implementation = {}
        self.layer_name_conv_implementation = {}

        self.model_compression = False
        self.layer_type_compression = {}
        self.layer_name_compression = {}

        self.trace_output = self.get_config_value('TraceOutput', False)

        self.pipeline_style = 'auto'
        self.pipeline_ii = None

        if 'WriterConfig' in self.config:
            self.writer_config = self.config['WriterConfig']
        else:
            self.writer_config = {
                'Namespace': None,
                'WriteWeightsTxt': True,
                'WriteTar': False,
                'TBOutputStream': 'both',
            }

        self._parse_hls_config()

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def get_project_name(self):
        return self.get_config_value('ProjectName')

    def get_project_dir(self):
        if self.get_config_value('ProjectDir') is not None:
            return self.get_config_value('ProjectDir')
        else:
            return self.get_config_value('ProjectName') + '_prj'

    def get_output_dir(self):
        return self.get_config_value('OutputDir')

    def get_layer_config_value(self, layer, key, default=None):
        hls_config = self.config['HLSConfig']

        name_config = hls_config.get('LayerName', {}).get(layer.name, None)
        if name_config is not None:
            return name_config.get(key, default)

        type_config = hls_config.get('LayerType', {}).get(layer.class_name, None)
        if type_config is not None:
            return type_config.get(key, default)

        model_config = hls_config.get('Model', None)
        if model_config is not None:
            return model_config.get(key, default)

        return default

    def get_layer_config(self, layer):
        hls_config = self.config['HLSConfig']
        layer_config = {}

        type_config = hls_config.get('LayerType', {}).get(layer.class_name, None)
        if type_config is not None:
            layer_config.update(type_config)

        name_config = hls_config.get('LayerName', {}).get(layer.name, None)
        if name_config is not None:
            layer_config.update(name_config)

        return layer_config

    def set_name_config(self, name, config):
        """sets hls_config["LayerName"][name] = config"""
        hls_config = self.config['HLSConfig']
        layer_config = hls_config.setdefault('LayerName', {})
        layer_config[name] = config

    def get_precision(self, layer, var='default'):
        precision = self.layer_name_precision.get(layer.name.lower() + '_' + var)
        type_name = layer.name.lower() + '_' + var + '_t'
        if precision is None:
            precision = self.layer_name_precision.get(layer.name.lower() + '_default')
            # I think it is better to keep these unique still to avoid inadvertent updates
            # type_name = layer.name.lower() + '_default_t'

        if precision is None:
            precision = self.layer_type_precision.get(layer.class_name.lower() + '_' + var)
            type_name = layer.class_name + '_' + var + '_t'
        if precision is None:
            precision = self.layer_type_precision.get(layer.class_name.lower() + '_default')
            type_name = layer.class_name + '_default_t'

        if precision is None:
            precision = self.model_precision.get(var)
            type_name = var + '_default_t'
        if precision is None:
            precision = self.model_precision.get('default')
            type_name = 'model_default_t'

        if precision is None:
            raise Exception(f'No precision for {layer.name}->{var} found and no default specified.')

        precision = self.backend.convert_precision_string(precision)

        return (precision, type_name)

    def get_bram_size(self, layer):
        bf = self.model_bf
        return bf

    def get_reuse_factor(self, layer):
        rf = self.layer_name_rf.get(layer.name.lower())
        if rf is None:
            rf = self.layer_type_rf.get(layer.class_name.lower())
        if rf is None:
            rf = self.model_rf

        if rf is None:
            raise Exception(f'No reuse factor for {layer.name} found and no default specified.')

        return rf

    def get_target_cycles(self, layer):
        targ_cycles = self.layer_name_targ_cycles.get(layer.name.lower())
        if targ_cycles is None:
            targ_cycles = self.layer_name_targ_cycles.get(layer.__class__.__name__.lower())
        if targ_cycles is None:
            targ_cycles = self.model_targ_cycles

        return targ_cycles

    def get_strategy(self, layer):
        strategy = self.layer_name_strategy.get(layer.name.lower())
        if strategy is None:
            strategy = self.layer_type_strategy.get(layer.class_name.lower())
        if strategy is None:
            strategy = self.model_strategy

        return strategy

    def get_conv_implementation(self, layer):
        conv_implementation = self.layer_name_conv_implementation.get(layer.name.lower())
        if conv_implementation is None:
            conv_implementation = self.layer_type_conv_implementation.get(layer.__class__.__name__.lower())
        if conv_implementation is None:
            conv_implementation = self.model_conv_implementation

        return conv_implementation

    def is_resource_strategy(self, layer):
        return self.get_strategy(layer).lower() == 'resource'

    def get_compression(self, layer):
        compression = self.layer_name_compression.get(layer.name.lower())
        if compression is None:
            compression = self.layer_type_compression.get(layer.class_name.lower())
        if compression is None:
            compression = self.model_compression

        return compression

    def parse_name_config(self, layer_name, layer_cfg):
        """This is used by _parse_hls_config below, but also in optimizers when a new layer config is created"""
        precision_cfg = layer_cfg.get('Precision')
        if isinstance(precision_cfg, dict):
            for var, precision in precision_cfg.items():
                self.layer_name_precision[layer_name.lower() + '_' + var] = precision
        else:
            self.layer_name_precision[layer_name.lower() + '_default'] = precision_cfg

        rf = layer_cfg.get('ReuseFactor')
        if rf is not None:
            self.layer_name_rf[layer_name.lower()] = rf

        targ_cycles = layer_cfg.get('TargetCycles')
        if targ_cycles is not None:
            self.layer_name_targ_cycles[layer_name.lower()] = targ_cycles

        strategy = layer_cfg.get('Strategy')
        if strategy is not None:
            self.layer_name_strategy[layer_name.lower()] = convert_to_snake_case(strategy)

        conv_implementation = layer_cfg.get('ConvImplementation')
        if conv_implementation is not None:
            self.layer_name_conv_implementation[layer_name.lower()] = conv_implementation

        compression = layer_cfg.get('Compression')
        if compression is not None:
            self.layer_name_compression[layer_name.lower()] = bool(compression)

    def get_writer_config(self):
        return self.writer_config

    def _parse_hls_config(self):
        hls_config = self.config['HLSConfig']

        self.flows = hls_config.get('Flows')
        if self.flows is None:
            self.flows = [self.backend.get_default_flow()]

        # TODO this is now effectively broken
        self.optimizers = hls_config.get('Optimizers')
        if 'SkipOptimizers' in hls_config:
            if self.optimizers is not None:
                raise Exception('Invalid optimizer configuration, please use either "Optimizers" or "SkipOptimizers".')
            skip_optimizers = hls_config.get('SkipOptimizers')
            selected_optimizers = get_available_passes()
            for opt in skip_optimizers:
                try:
                    selected_optimizers.remove(opt)
                except ValueError:
                    pass
            self.optimizers = selected_optimizers

        model_cfg = hls_config.get('Model')
        if model_cfg is not None:
            precision_cfg = model_cfg.get('Precision')
            if precision_cfg is not None:
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.model_precision[var] = precision
                else:
                    self.model_precision['default'] = precision_cfg  # Default precision for everything

            self.model_bf = model_cfg.get('BramFactor', np.inf)  # Weight threshold to be external BRAM
            self.model_rf = model_cfg.get('ReuseFactor')
            self.model_targ_cycles = model_cfg.get('TargetCycles')
            self.model_conv_implementation = model_cfg.get('ConvImplementation', 'LineBuffer')
            self.model_strategy = convert_to_snake_case(model_cfg.get('Strategy', 'Latency'))
            self.model_compression = bool(model_cfg.get('Compression', 0))
            self.pipeline_style = model_cfg.get('PipelineStyle', 'auto')
            self.pipeline_ii = model_cfg.get('PipelineInterval', None)

        layer_type_cfg = hls_config.get('LayerType')
        if layer_type_cfg is not None:
            for layer_type, layer_cfg in layer_type_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_type_precision[layer_type.lower() + '_' + var] = precision
                else:
                    self.layer_type_precision[layer_type.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_type_rf[layer_type.lower()] = rf

                targ_cycles = layer_cfg.get('TargetCycles')
                if targ_cycles is not None:
                    self.layer_type_targ_cycles[layer_type.lower()] = targ_cycles

                strategy = layer_cfg.get('Strategy')
                if strategy is not None:
                    self.layer_type_strategy[layer_type.lower()] = convert_to_snake_case(strategy)

                conv_implementation = layer_cfg.get('ConvImplementation')
                if conv_implementation is not None:
                    self.layer_type_conv_implementation[layer_type.lower()] = conv_implementation

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_type_compression[layer_type.lower()] = bool(compression)

        layer_name_cfg = hls_config.get('LayerName')
        if layer_name_cfg is not None:
            for layer_name, layer_cfg in layer_name_cfg.items():
                self.parse_name_config(layer_name, layer_cfg)

    def serialize(self):
        state = {}

        config = self.config.copy()
        config.pop('KerasModel', None)
        config.pop('OnnxModel', None)
        config.pop('PytorchModel', None)

        # Much of this may not be needed and is already in 'config' dict but is kept here to be sure
        state['config'] = config
        state['model_precision'] = self.model_precision.copy()
        state['layer_type_precision'] = self.layer_type_precision.copy()
        state['layer_name_precision'] = self.layer_name_precision.copy()

        state['model_rf'] = self.model_rf
        state['layer_type_rf'] = self.layer_type_rf.copy()
        state['layer_name_rf'] = self.layer_name_rf.copy()

        state['model_targ_cycles'] = self.model_targ_cycles
        state['layer_type_targ_cycles'] = self.layer_type_targ_cycles.copy()
        state['layer_name_targ_cycles'] = self.layer_name_targ_cycles.copy()

        state['model_strategy'] = self.model_strategy
        state['layer_type_strategy'] = self.layer_type_strategy.copy()
        state['layer_name_strategy'] = self.layer_name_strategy.copy()

        state['model_conv_implementation'] = self.model_conv_implementation
        state['layer_type_conv_implementation'] = self.layer_type_conv_implementation.copy()
        state['layer_name_conv_implementation'] = self.layer_name_conv_implementation.copy()

        state['model_compression'] = self.model_compression
        state['layer_type_compression'] = self.layer_type_compression.copy()
        state['layer_name_compression'] = self.layer_name_compression.copy()

        state['trace_output'] = self.trace_output
        state['pipeline_style'] = self.pipeline_style
        state['pipeline_ii'] = self.pipeline_ii
        state['writer_config'] = self.writer_config.copy()
        state['flows'] = self.flows.copy()
        state['optimizers'] = self.optimizers.copy() if self.optimizers is not None else None
        state['model_bf'] = self.model_bf

        return state

    @classmethod
    def deserialize(cls, state):
        config = cls(state['config'])

        config.model_precision = state['model_precision']
        config.layer_type_precision = state['layer_type_precision']
        config.layer_name_precision = state['layer_name_precision']

        config.model_rf = state['model_rf']
        config.layer_type_rf = state['layer_type_rf']
        config.layer_name_rf = state['layer_name_rf']

        config.model_targ_cycles = state['model_targ_cycles']
        config.layer_type_targ_cycles = state['layer_type_targ_cycles']
        config.layer_name_targ_cycles = state['layer_name_targ_cycles']

        config.model_strategy = state['model_strategy']
        config.layer_type_strategy = state['layer_type_strategy']
        config.layer_name_strategy = state['layer_name_strategy']

        config.model_conv_implementation = state['model_conv_implementation']
        config.layer_type_conv_implementation = state['layer_type_conv_implementation']
        config.layer_name_conv_implementation = state['layer_name_conv_implementation']

        config.model_compression = state['model_compression']
        config.layer_type_compression = state['layer_type_compression']
        config.layer_name_compression = state['layer_name_compression']

        config.trace_output = state['trace_output']
        config.pipeline_style = state['pipeline_style']
        config.pipeline_ii = state['pipeline_ii']
        config.writer_config = state['writer_config']
        config.flows = state['flows']
        config.optimizers = state['optimizers']
        config.model_bf = state['model_bf']

        return config


class ModelGraph(Serializable):
    """The ModelGraph represents the network that is being processed by hls4ml.

    Args:
        config (dict):  The configuration dictionary
        layer_list (list(dict)):  The list contains a dictionary for each input layer
        inputs (list, optional):  The inputs to the model. If None, determined from layer_list
        outputs (list, optional):  The outputs to the model. If None, determined from layer_list
    """

    def __init__(self, config, inputs=None, outputs=None, initial_index=0):
        self.config = config
        self.inputs = inputs
        self.outputs = outputs
        self.graph = OrderedDict()
        self._applied_flows = []  # keep track of the applied flows
        self.index = initial_index
        self.output_vars = {}
        self._top_function_lib = None

    @classmethod
    def from_layer_list(cls, config_dict, layer_list, inputs=None, outputs=None, initial_index=0):
        def _find_output_variable_names(layer_list, layer_names):
            """Given a list of all layers, and a list input/output names, find the names of their outputs that will be used
            as the name of the output variables."""
            inout_nodes = []
            for layer_name in layer_names:
                for node in layer_list:
                    if node['name'] == layer_name:
                        inout_nodes.append(node)
            all_node_output_names = [node['outputs'] if 'outputs' in node else [node['name']] for node in inout_nodes]
            return [output for node_output_names in all_node_output_names for output in node_output_names]  # to flatten

        config = HLSConfig(config_dict)

        # If not provided, assumes layer_list[0] is the input layer, and layer_list[-1] is output layer
        # Note, these are actually the variable names, which may differ from the layer name
        input_layers = inputs if inputs is not None else [layer_list[0]['name']]
        output_layers = outputs if outputs is not None else [layer_list[-1]['name']]
        input_names = _find_output_variable_names(layer_list, input_layers)
        if input_names != input_layers:
            raise RuntimeError(
                "Currently only support the case when input variables and input layer names match\n"
                + f"Input layers = {input_layers}, input_vars = {input_names}"
            )
        output_names = _find_output_variable_names(layer_list, output_layers)

        model = cls(config, input_names, output_names, initial_index)
        model._make_graph(layer_list)
        for flow in model.config.flows:
            model.apply_flow(flow)

        model.config.config['InputShapes'] = {}
        for input_var in model.get_input_variables():
            model.config.config['InputShapes'][input_var.name] = list(input_var.shape)
        model.config.config['OutputShapes'] = {}
        for output_var in model.get_output_variables():
            model.config.config['OutputShapes'][output_var.name] = list(output_var.shape)

        return model

    @classmethod
    def from_saved_state(cls, config, graph_state_dict):
        model = cls(config, graph_state_dict['inputs'], graph_state_dict['outputs'])
        model._applied_flows = graph_state_dict['applied_flows']
        model.index = graph_state_dict['index']

        return model

    def _make_graph(self, layer_list):
        for layer in layer_list:
            kind = layer['class_name']
            name = layer['name']
            inputs = layer.get('inputs', [])
            outputs = layer.get('outputs', [])
            if kind in ['InputLayer', 'Input']:
                inputs = ['input']
            elif len(inputs) == 0:
                inputs = [next(reversed(self.graph), 'input')]
            if len(outputs) == 0:
                outputs = [name]

            self.graph[name] = self.make_node(kind, name, layer, inputs, outputs)

    def apply_flow(self, flow, reapply='single'):
        """Applies a flow (a collection of optimizers).

        Args:
            flow (str): The name of the flow to apply
            reapply (str, optional): Determines the action to take if the flow and its requirements have already been
                applied. Possible values are:
                - 'all': Apply the flow and all its requirements.
                - 'single': Apply only the given flow, but skip the already applied requirements.
                - 'none': Skip applying the flow.
                Defaults to 'single'.
        """

        def all_applied_flows():
            applied_flows = {}

            for flow_group in self._applied_flows:
                applied_flows.update({flow: set() for flow in flow_group.keys()})

            return applied_flows

        assert reapply in ['all', 'single', 'none']

        if reapply == 'all':
            applied_flows = {}
        elif reapply == 'single':
            applied_flows = all_applied_flows()
            applied_flows.pop(flow, None)
        else:  # reapply == 'none'
            applied_flows = all_applied_flows()
            if flow in applied_flows:
                return

        self._applied_flows.append(applied_flows)
        self._apply_sub_flow(flow, applied_flows)

    def _apply_sub_flow(self, flow_name, applied_flows):
        if flow_name in applied_flows:
            return
        flow = get_flow(flow_name)

        for sub_flow in flow.requires:
            if sub_flow not in applied_flows.keys():
                self._apply_sub_flow(sub_flow, applied_flows)

        if len(flow.optimizers) > 0:
            applied_passes = optimize_model(self, flow.optimizers)
        else:
            applied_passes = set()
        applied_flows[flow.name] = applied_passes

    def make_node(self, kind, name, attributes, inputs, outputs=None, initialize=True):
        """Make a new node not connected to the model graph.

        The 'kind' should be a valid layer registered with `register_layer`. If no outputs
        are specified, a default output named the same as the node will be created. The
        returned node should be added to the graph with `insert_node` or `replace_node`
        functions.

        Args:
            kind (type or str): Type of node to add
            name (str): Name of the node
            attributes (dict): Initial set of attributes required to construct the node (Layer)
            inputs (list): List of inputs to the layer
            outputs (list, optional): The optional list of named outputs of the node
            initialize (bool, optional): Whether to call the `initialize()` of a layer. Defaults to True.
                Set to False during deserialization.

        Raises:
            Exception: If an attempt to insert a node with multiple inputs is made or if
                `before` does not specify a correct node in sequence.

        Returns:
            Layer: The node created.
        """

        if isinstance(kind, str):
            if kind not in layer_map:
                raise Exception(f'Layer {kind} not found in registry.')
            layer_cls = layer_map[kind]
        else:
            if kind not in layer_map.values():
                raise Exception(f'Layer {kind} not found in registry.')
            layer_cls = kind

        if self.config.backend is not None:
            layer_cls = self.config.backend.create_layer_class(layer_cls)
        node = layer_cls(self, name, attributes, inputs, outputs, initialize)
        for o in node.outputs:
            out_var = node.get_output_variable(output_name=o)
            if len(self.outputs) == 1 and o in self.outputs:
                out_var.type.name = 'result_t'
            self.output_vars[o] = out_var
        return node

    def insert_node(self, node, before=None, input_idx=0):
        """Insert a new node into the model graph.

        The node to be inserted should be created with `make_node()` function. The optional
        parameter `before` can be used to specify the node that follows in case of ambiguities.

        Args:
            node (Layer): Node to insert
            before (Layer, optional): The next node in sequence before which a
                new node should be inserted.
            input_idx (int, optional): If the next node takes multiple inputs, the input index
        Raises:
            Exception: If an attempt to insert a node with multiple inputs is made or if
                `before` does not specify a correct node in sequence.

        """
        if len(node.inputs) > 1:
            raise Exception('Cannot insert a node with more than one input (for now).')

        prev_node = node.get_input_node(node.inputs[0])
        next_nodes = []
        for x in self.graph.values():
            overlap = [value for value in x.inputs if value in prev_node.outputs]
            if overlap:
                next_nodes.append(x)

        if before is None:
            next_node = next((x for x in self.graph.values() if x.inputs and x.inputs[0] in prev_node.outputs), None)
        else:
            if before not in next_nodes:
                raise Exception(
                    'Cannot insert a node {} before {} (candidates: {}).'.format(
                        node.name, before.name, ','.join([n.name for n in next_nodes])
                    )
                )
            next_node = before

        if next_node is not None:
            next_node.inputs[input_idx] = node.outputs[0]
        else:
            self.outputs = [node.outputs[0] if name == prev_node.outputs[0] else name for name in self.outputs]

        new_graph = OrderedDict()
        for k, v in self.graph.items():
            new_graph[k] = v
            if k == prev_node.name:
                new_graph[node.name] = node

        self.graph = new_graph

    def remove_node(self, node):
        """Removes a node from the graph.

        By default, this function connects the outputs of the previous
        node to the inputs of the next node. If the removed node has multiple
        input/output tensors, an exception is raised.

        Args:
            node (Layer): The node to remove.

        Raises:
            Exception: If an attempt is made to remove a node with
            multiple inputs/outputs.
        """

        inputs = [inp for inp in node.inputs if inp]
        outputs = [outp for outp in node.outputs if outp]

        if len(inputs) > 1 or len(outputs) > 1:
            raise Exception('Cannot delete a node with multiple inputs/outputs')

        if len(outputs) == 1 and len(inputs) == 1:

            # Connect inputs -> $outputs
            if node.outputs[0] in self.outputs:
                msg = f'Remove leaf node {node.name} will connect its input node {inputs[0]} to output, but it already is.'
                assert inputs[0] not in self.outputs, msg
                self.outputs = [inputs[0] if name == node.outputs[0] else name for name in self.outputs]

            inp_var = node.get_input_variable()
            out_var = node.get_output_variable()

            # fmt: off
            assert (np.prod(inp_var.shape) == np.prod(out_var.shape)), \
                f'Input and output shapes do not match for {node.name}: {inp_var.shape} -> {out_var.shape}'
            # fmt: on

            next_nodes = [x for x in self.graph.values() if node.outputs[0] in x.inputs]
            for next_node in next_nodes:
                # Connect inputs -> next
                for i, nxt_inp in enumerate(next_node.inputs):
                    if outputs[0] == nxt_inp:
                        next_node.inputs[i] = inputs[0]

        del self.output_vars[node.outputs[0]]
        del self.graph[node.name]

    def replace_node(self, old_node, new_node):
        """Replace an existing node in the graph with a new one.

        Args:
            old_node (Layer): The node to replace
            new_node (Layer): The new node

        """

        # fmt: off
        assert len(new_node.inputs) == len(old_node.inputs), \
            f'{new_node.name} and {old_node.name} have different number of inputs'
        assert len(new_node.outputs) == len(old_node.outputs), \
            f'{new_node.name} and {old_node.name} have different number of outputs'
        # fmt: on

        repl = {old_name: new_name for old_name, new_name in zip(old_node.outputs, new_node.outputs)}
        repl.update({old_name: new_name for old_name, new_name in zip(old_node.inputs, new_node.inputs)})

        for node in self.graph.values():
            for i, n in enumerate(node.inputs):
                if n in repl:
                    node.inputs[i] = repl[n]
            for i, n in enumerate(node.outputs):
                if n in repl:
                    node.outputs[i] = repl[n]

        self.graph = OrderedDict((new_node.name, new_node) if k == old_node.name else (k, v) for k, v in self.graph.items())

        old_name = old_node.name
        if old_name in self.outputs:
            new_name = new_node.name
            self.outputs = [new_name if name == old_name else name for name in self.outputs]

    def split_node(self, old_node, new_node1, new_node2):
        """Replace an existing node in the graph with two nodes in sequence.

        Args:
            old_node (Layer): The node to replace
            new_node1 (Layer): The first new node in sequence
            new_node2 (Layer): The second new node in sequence

        """

        # fmt: off
        assert len(new_node1.inputs) == len(old_node.inputs), \
            f'{new_node1.name} and {old_node.name} have different number of inputs'
        assert len(new_node2.outputs) == len(old_node.outputs), \
            f'{new_node2.name} and {old_node.name} have different number of outputs'
        # fmt: on

        repl = {old_name: new_name for old_name, new_name in zip(old_node.outputs, new_node2.outputs)}
        repl.update({old_name: new_name for old_name, new_name in zip(old_node.inputs, new_node1.inputs)})

        for node in self.graph.values():
            for i, n in enumerate(node.inputs):
                if n in repl:
                    node.inputs[i] = repl[n]
            for i, n in enumerate(node.outputs):
                if n in repl:
                    node.outputs[i] = repl[n]

        new_graph = OrderedDict()
        for key, value in self.graph.items():
            if key == old_node.name:
                new_graph[new_node1.name] = new_node1
                new_graph[new_node2.name] = new_node2
            else:
                new_graph[key] = value
        self.graph = new_graph

        if old_node.name in self.outputs:
            self.outputs = [new_node2.name if name == old_node.name else name for name in self.outputs]

    def next_layer(self):
        self.index += 1
        return self.index

    def get_layers(self):
        return self.graph.values()

    def get_input_variables(self):
        variables = []
        for inp in self.inputs:
            variables.append(self.graph[inp].get_output_variable())
        return variables

    def register_output_variable(self, out_name, variable):
        if len(self.outputs) == 1 and out_name in self.outputs:
            variable.type.name = 'result_t'
        self.output_vars[out_name] = variable

    def get_output_variables(self):
        variables = []
        for out in self.outputs:
            variables.append(self.output_vars[out])
        return variables

    def get_layer_output_variable(self, output_name):
        return self.output_vars.get(output_name, None)

    def get_weight_variables(self):
        variables = []
        for layer in self.get_layers():
            weights = layer.get_weights()
            variables.extend(weights)

        return variables

    def write(self):
        """Write the generated project to disk.

        This function converts the model to C++ and writes the generated files in the output
        directory specified in the `config`.
        """

        self.config.backend.write(self)

    def compile(self):
        """Compile the generated project and link the library into current environment.

        Users should call this function if they want to use `predict` functionality for simulation.
        """
        self.write()
        self._compile()

    def _compile(self):
        lib_name = self.config.backend.compile(self)
        if self._top_function_lib is not None:
            if platform.system() == "Linux":
                libdl_libs = ['libdl.so', 'libdl.so.2']
                for libdl in libdl_libs:
                    try:
                        dlclose_func = ctypes.CDLL(libdl).dlclose
                        break
                    except Exception:
                        continue
            elif platform.system() == "Darwin":
                dlclose_func = ctypes.CDLL('libc.dylib').dlclose

            dlclose_func.argtypes = [ctypes.c_void_p]
            dlclose_func.restype = ctypes.c_int
            dlclose_func(self._top_function_lib._handle)
        self._top_function_lib = ctypes.cdll.LoadLibrary(lib_name)

    def _get_top_function(self, x):
        if self._top_function_lib is None:
            raise Exception('Model not compiled')
        if len(self.get_input_variables()) == 1:
            xlist = [x]
        else:
            xlist = x
        n_outputs = len(self.get_output_variables())

        for xi in xlist:
            if not isinstance(xi, np.ndarray):
                raise Exception(f'Expected numpy.ndarray, but got {type(x)}')
            if not xi.flags['C_CONTIGUOUS']:
                raise Exception('Array must be c_contiguous, try using numpy.ascontiguousarray(x)')

        x0 = xlist[0]
        if x0.dtype in [np.single, np.float32]:
            top_function = getattr(self._top_function_lib, self.config.get_project_name() + '_float')
            ctype = ctypes.c_float
        elif x0.dtype in [np.double, np.float64]:
            top_function = getattr(self._top_function_lib, self.config.get_project_name() + '_double')
            ctype = ctypes.c_double
        else:
            raise Exception(
                'Invalid type ({}) of numpy array. Supported types are: single, float32, double, float64, float_.'.format(
                    x0.dtype
                )
            )

        top_function.restype = None
        top_function.argtypes = [npc.ndpointer(ctype, flags="C_CONTIGUOUS") for i in range(len(xlist) + n_outputs)]

        return top_function, ctype

    def _compute_n_samples(self, x):
        if len(self.get_input_variables()) == 1:
            xlist = [x]
        else:
            xlist = x
        n_samples = []
        for i, xi in enumerate(xlist):
            expected_size = self.get_input_variables()[i].size()
            x_size = np.prod(xi.shape)
            n_sample, rem = divmod(x_size, expected_size)
            if rem != 0:
                raise Exception(f'Input size mismatch, got {x_size.shape}, expected {self.get_input_variables()[i].shape}')
            n_samples.append(n_sample)

        if not all([n_samples[i] == n_samples[i + 1] for i in range(len(xlist) - 1)]):
            raise Exception('Input size mismatch, not all inputs match')

        return int(n_sample)

    def predict(self, x):
        top_function, ctype = self._get_top_function(x)
        n_samples = self._compute_n_samples(x)
        n_inputs = len(self.get_input_variables())
        n_outputs = len(self.get_output_variables())

        output = []
        if n_samples == 1 and n_inputs == 1:
            x = [x]

        for i in range(n_samples):
            predictions = [np.zeros(yj.size(), dtype=ctype) for yj in self.get_output_variables()]
            if n_inputs == 1:
                inp = [np.asarray(x[i])]
            else:
                inp = [np.asarray(xj[i]) for xj in x]
            inp = [np.ascontiguousarray(_inp) for _inp in inp]

            top_function(*inp, *predictions)
            output.append(predictions)

        # Convert to list of numpy arrays (one for each output)
        output = [np.asarray([output[i_sample][i_output] for i_sample in range(n_samples)]) for i_output in range(n_outputs)]

        if n_samples == 1 and n_outputs == 1:
            return output[0][0]
        elif n_outputs == 1:
            return output[0]
        elif n_samples == 1:
            return [output_i[0] for output_i in output]
        else:
            return output

    def trace(self, x):
        print(f'Recompiling {self.config.get_project_name()} with tracing')
        self.config.trace_output = True
        self.compile()

        top_function, ctype = self._get_top_function(x)
        n_samples = self._compute_n_samples(x)
        n_inputs = len(self.get_input_variables())
        n_outputs = len(self.get_output_variables())

        class TraceData(ctypes.Structure):
            _fields_ = [('name', ctypes.c_char_p), ('data', ctypes.c_void_p)]

        trace_output = {}
        layer_sizes = {}
        n_traced = 0
        for layer in self.get_layers():
            if layer.get_attr('function_cpp', None) and layer.get_attr('trace', False):
                n_traced += len(layer.get_variables())
                trace_output[layer.name] = []
                layer_sizes[layer.name] = layer.get_output_variable().shape

        collect_func = self._top_function_lib.collect_trace_output
        collect_func.argtypes = [ctypes.POINTER(TraceData)]
        collect_func.restype = None
        trace_data = (TraceData * n_traced)()

        alloc_func = self._top_function_lib.allocate_trace_storage
        alloc_func.argtypes = [ctypes.c_size_t]
        alloc_func.restype = None

        free_func = self._top_function_lib.free_trace_storage
        free_func.argtypes = None
        free_func.restype = None

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir() + '/firmware')

        output = []
        if n_samples == 1 and n_inputs == 1:
            x = [x]

        try:
            alloc_func(ctypes.sizeof(ctype))

            for i in range(n_samples):
                predictions = [np.zeros(yj.size(), dtype=ctype) for yj in self.get_output_variables()]
                if n_inputs == 1:
                    inp = [np.asarray(x[i])]
                else:
                    inp = [np.asarray(xj[i]) for xj in x]
                argtuple = inp
                argtuple += predictions
                argtuple = tuple(argtuple)
                top_function(*argtuple)
                output.append(predictions)
                collect_func(trace_data)
                for trace in trace_data:
                    layer_name = str(trace.name, 'utf-8')
                    layer_data = ctypes.cast(trace.data, ctypes.POINTER(ctype))
                    np_array = np.ctypeslib.as_array(layer_data, shape=layer_sizes[layer_name])
                    trace_output[layer_name].append(np.copy(np_array))

            for key in trace_output.keys():
                trace_output[key] = np.asarray(trace_output[key])

            # Convert to list of numpy arrays (one for each output)
            output = [
                np.asarray([output[i_sample][i_output] for i_sample in range(n_samples)]) for i_output in range(n_outputs)
            ]

            free_func()
        finally:
            os.chdir(curr_dir)

        if n_samples == 1 and n_outputs == 1:
            return output[0][0], trace_output
        elif n_outputs == 1:
            return output[0], trace_output
        elif n_samples == 1:
            return [output_i[0] for output_i in output], trace_output
        else:
            return output, trace_output

    def build(self, **kwargs):
        """Builds the generated project using HLS compiler.

        Please see the `build()` function of backends for a list of possible arguments.
        """
        if not os.path.exists(self.config.get_output_dir()):
            # Assume the project wasn't written before
            self.write()

        return self.config.backend.build(self, **kwargs)

    def serialize(self):
        applied_flows = []
        for flow_group in self._applied_flows:
            flow_cpy = {}
            for flow_name, opt_set in flow_group.items():
                flow_cpy[flow_name] = list(opt_set)
            applied_flows.append(flow_cpy)
        state = {
            'inputs': self.inputs.copy(),
            'outputs': self.outputs.copy(),
            'index': self.index,
            'applied_flows': applied_flows,
        }

        return state

    @classmethod
    def deserialize(cls, state):
        raise Exception(
            f'{cls.__name__} is not intended to be deserialized directly. Use {cls.__name__}.from_saved_state instead.'
        )

    def save(self, file_path):
        """Saves the ModelGraph to a file.

        See `hls4ml.utils.serialization.serialize_model` for details on the file format.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        from hls4ml.utils.serialization import serialize_model

        serialize_model(self, file_path)


class MultiModelGraph:
    def __init__(self, graphs: list[ModelGraph]):
        """
        Create a stitched model from pre-optimized subgraphs.
        """
        self.graphs = graphs
        self._initialize_config(self.graphs[0])
        self._bind_modelgraph_methods()
        self._initialize_io_attributes(self.graphs)

    @classmethod
    def from_model_graph(cls, base_model: ModelGraph, split_before_layers: list[str]):
        """
        Create a MultiModelGraph by splitting a base ModelGraph at specified layer names,
        each initiating a subgraph.
        """
        cls._validate_split_points(base_model, split_before_layers)
        all_nodes = list(base_model.graph.values())
        layer_names = [node.name for node in all_nodes]
        split_indices = sorted(layer_names.index(s) for s in split_before_layers)
        bounds = [0] + split_indices + [len(all_nodes)]
        node_slices: list[list] = [all_nodes[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]

        base_input_layer = base_model.graph[base_model.inputs[0]]
        input_layer_kind = base_input_layer.attributes['class_name']
        next_index = max(n.index for n in all_nodes)

        subgraphs: list['ModelGraph'] = []
        for idx, slice_ in enumerate(node_slices):
            cfg_copy = copy.copy(base_model.config)
            cfg_copy.config = copy.copy(base_model.config.config)
            cfg_copy.config['ProjectName'] = f'{base_model.config.get_project_name()}_graph{idx + 1}'
            cfg_copy.config['OutputDir'] = os.path.join(base_model.config.get_output_dir(), f'graph{idx + 1}')

            subgraph = base_model.__class__(cfg_copy, inputs=[], outputs=[])
            graph_dict = OrderedDict()

            if idx > 0:
                next_index += 1
                input_layer = cls._create_input_node(subgraph, slice_[0], input_layer_kind, next_index)
                graph_dict[input_layer.name] = input_layer
                slice_[0].inputs = input_layer.outputs
            else:
                input_layer = base_input_layer

            for node in slice_:
                node.model = subgraph  # fix for layer.model.get_layer_output_variable()
                for out_name in node.outputs:
                    subgraph.output_vars[out_name] = base_model.output_vars[out_name]
                graph_dict[node.name] = node

            subgraph.graph = graph_dict
            subgraph.inputs = input_layer.outputs if idx > 0 else base_model.inputs
            subgraph.outputs = slice_[-1].outputs if idx < len(node_slices) - 1 else base_model.outputs
            subgraph._applied_flows = base_model._applied_flows

            # NOTE might need to examine other subgraph-related flows (i.e., fifo_optimizer)
            subgraph.apply_flow('vivado:specific_types')
            subgraph.apply_flow('vitis:apply_templates')

            input_var = subgraph.output_vars[input_layer.name]
            if getattr(input_var, 'pragma', None) == 'reshape':
                input_var.pragma = 'partition'  # NOTE required for subgraph stitching; subject to refinement

            subgraphs.append(subgraph)

        return cls(subgraphs)

    @staticmethod
    def _validate_split_points(model: ModelGraph, split_names: list[str]):
        if not split_names:
            raise ValueError('No split layer names provided.')

        nodes = list(model.graph.values())
        model_layers = [node.name for node in nodes]
        for name in split_names:
            node = model.graph[name]
            if name not in model_layers:
                raise ValueError(f"Split layer '{name}' not found in the model.")
            if len(node.inputs) > 1:
                raise ValueError(f"Cannot split at layer '{name}' (multiple inputs detected).")
            if model.graph[node.inputs[0]].class_name == 'Reshape' or node.class_name == 'Reshape':
                raise ValueError(f"Cannot split at '{name}': Reshape layer found in this or previous layer.")

    @staticmethod
    def _create_input_node(model, next_node, kind, index):
        layer_name = f'{next_node.name}_input'
        attrs = {
            'name': layer_name,
            'class_name': kind,
            'data_format': 'channels_last',
            'input_shape': next_node.get_input_variable().shape,
        }
        model.index = index
        node = model.make_node(kind, layer_name, attrs, [layer_name], [layer_name], initialize=True)
        model.output_vars[layer_name].type.precision = next_node.get_input_variable().type.precision
        return node

    def _initialize_config(self, first_graph):
        self.config = copy.copy(first_graph.config)
        keys_to_deepcopy = ['ProjectName', 'OutputDir']
        self.config.config = {
            k: copy.deepcopy(first_graph.config.config[k]) if k in keys_to_deepcopy else first_graph.config.config[k]
            for k in first_graph.config.config
        }
        self._update_project_config(first_graph)
        self.backend = first_graph.config.backend

    def _bind_modelgraph_methods(self):
        # Bind necessary ModelGraph methods to this instance
        self._compile = ModelGraph._compile.__get__(self, MultiModelGraph)
        self.get_output_variables = ModelGraph.get_output_variables.__get__(self, MultiModelGraph)
        self._compute_n_samples = ModelGraph._compute_n_samples.__get__(self, MultiModelGraph)
        self._get_top_function = ModelGraph._get_top_function.__get__(self, MultiModelGraph)
        self._predict = ModelGraph.predict.__get__(self, MultiModelGraph)

    def _initialize_io_attributes(self, graphs):
        self.graph_reports = None
        self._top_function_lib = None
        self.inputs = graphs[0].inputs
        self.outputs = graphs[-1].outputs
        self.output_vars = {k: v for graph in graphs for k, v in graph.output_vars.items()}

    def _update_project_config(self, first_graph):
        original_project_name = first_graph.config.get_project_name().partition('_graph')[0]
        self.config.config['ProjectName'] = f"{original_project_name}_stitched"
        self.config.config['OriginalProjectName'] = original_project_name
        original_output_dir = first_graph.config.get_output_dir().partition('/graph')[0]
        self.config.config['OutputDir'] = os.path.join(original_output_dir, 'stitched')
        self.config.config['StitchedProjectName'] = 'vivado_stitched_design'

    def __getitem__(self, index):
        return self.graphs[index]

    def parse_nn_config(self):
        nn_config = {"inputs": [], "outputs": []}
        nn_config['OutputDir'] = self.config.config['OutputDir']
        nn_config['StitchedProjectName'] = self.config.config['StitchedProjectName']
        nn_config['OriginalProjectName'] = self.config.config['OriginalProjectName']

        # Parse layers (inputs and outputs)
        for graph, io_type in [(self.graphs[0], "inputs"), (self.graphs[-1], "outputs")]:
            for layer in getattr(graph, io_type):
                if layer in graph.output_vars:
                    total_bits = 1
                    [total_bits := total_bits * num for num in graph.output_vars[layer].shape]
                    pragma = graph.output_vars[layer].pragma
                    layer_pragma, fifo_depth = self._get_pragma_details(pragma)
                    if total_bits % fifo_depth != 0:
                        raise ValueError('Division of total_bits by fifo_depth does not result in a remainder of zero.')
                    batch_size = total_bits // fifo_depth
                    precision = graph.output_vars[layer].type.precision
                    nn_config[io_type].append(
                        {
                            'name': graph.output_vars[layer].name,
                            'pragma': layer_pragma,
                            'integer_bits': int(precision.integer),
                            'fractional_bits': int(precision.fractional),
                            'signed': int(precision.signed),
                            'fifo_depth': int(fifo_depth),
                            'batch_size': int(batch_size),
                        }
                    )

        return nn_config

    def build(
        self,
        export=True,
        stitch_design=False,
        sim_stitched_design=False,
        export_stitched_design=False,
        max_workers=None,
        **kwargs,
    ):
        """
        Builds all ModelGraph instances in parallel, with optional stitching, simulation and export.

        Args:
            export (bool): If True, export each subgraph as an IP (must be true for stiching design).
            stitch_design (bool): If True, create a Vivado stitched design project.
            sim_stitched_design (bool): If True, simulate the stitched design using Verilog in RTL level.
            export_stitched_design (bool): If True, export stitched design as a single IP.
            max_workers (int, optional): Maximum number of threads to use for parallel subgraph synthesis.
            **kwargs: Additional arguments passed to each subgraph's '.build()' method.
        Returns:
            Report from each subgraph's build and, if stitching was performed, a combined report of the stitched design.
        """
        if (stitch_design or sim_stitched_design or export_stitched_design) and not export:
            raise ValueError(
                'You cannot enable stitch_design, sim_stitched_design, or export_stitched_design without having export=True.'
            )
        if (sim_stitched_design or export_stitched_design) and not stitch_design:
            raise ValueError('You cannot simulate or export a stitched design without enabling stitch_design.')

        build_results = {}
        status = {}
        status_lock = threading.Lock()

        for idx, _ in enumerate(self.graphs, start=1):
            status[f'graph{idx}'] = 'Pending'

        def build_wrapper(idx, g, **kwargs):
            graph_name = f'graph{idx}'
            with status_lock:
                status[graph_name] = 'Running'
                self._print_status(status)
            try:
                result = g.build(log_to_stdout=False, export=export, **kwargs)
                with status_lock:
                    status[graph_name] = 'Completed'
                    self._print_status(status)
                return result
            except Exception as exc:
                with status_lock:
                    status[graph_name] = 'Failed'
                    self._print_status(status)
                raise exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(build_wrapper, idx, g, **kwargs): idx for idx, g in enumerate(self.graphs, start=1)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                graph_name = f'graph{idx}'
                try:
                    result = future.result()
                    build_results[graph_name] = result
                except Exception as exc:
                    build_results[graph_name] = None
                    print(f'Error while building {graph_name}: {exc}')

        self.graph_reports = build_results

        if stitch_design or sim_stitched_design or export_stitched_design:
            failed_graphs = [name for name, report in build_results.items() if report is None]
            if failed_graphs:
                print(f"Skipping stitching. Build failed for the following subgraphs: {', '.join(failed_graphs)}")
                return self.graph_reports

            self._replace_logos()
            self._assert_consistent_pragmas()
            self.nn_config = self.parse_nn_config()
            stitched_report = self.backend.build_stitched_design(
                self,
                stitch_design=stitch_design,
                sim_stitched_design=sim_stitched_design,
                export_stitched_design=export_stitched_design,
                graph_reports=self.graph_reports,
            )
            return stitched_report

        return self.graph_reports

    def write(self):
        for g in self.graphs:
            g.write()
        self.nn_config = self.parse_nn_config()
        self.config.config['Stamp'] = self._make_stamp()
        # Bypass VitisWriter and invoke write_hls directly from VivadoWriter
        super(self.backend.writer.__class__, self.backend.writer).write_hls(self, is_multigraph=True)

    def compile(self):
        self.write()
        self._compile()

    def predict(self, x, sim='csim'):
        if sim == 'csim':
            return self._predict(x)
        elif sim == 'rtl':
            self.nn_config = self.parse_nn_config()
            assert (
                np.prod(x.shape) == self.nn_config['inputs'][0]['fifo_depth'] * self.nn_config['inputs'][0]['batch_size']
            ), 'Only single batch supported for stitched simulation.'
            stitched_report = self.backend.build_stitched_design(
                self,
                stitch_design=False,
                sim_stitched_design=True,
                export_stitched_design=False,
                graph_reports=self.graph_reports,
                simulation_input_data=x,
            )

            results = stitched_report.get('BehavSimResults', [])
            if isinstance(results, np.ndarray):
                return results.astype(np.float32) if x.dtype in [np.single, np.float32] else results.astype(np.float64)
            elif isinstance(results, list):
                return [
                    arr.astype(np.float32) if x.dtype in [np.single, np.float32] else arr.astype(np.float64)
                    for arr in results
                ]
            else:
                return results
        else:
            print('Unknown simulation option given.')

    def trace(self, x):
        raise NotImplementedError('Trace function has not been implemented yet for MultiModelGraph.')

    def write_tb_inputs(self, x, folder_path):
        """
        Dump inputs (for Verilog testbench) via the C++ bridge functions:
        dump_tb_inputs_float
        dump_tb_inputs_double
        """
        if self._top_function_lib is None:
            self.compile()

        if isinstance(x, (list, tuple)):
            xlist = list(x)
        else:
            xlist = [x]

        first = xlist[0]
        if first.dtype in [np.single, np.float32]:
            fn_name = 'dump_tb_inputs_float'
            ctype = ctypes.c_float
        elif first.dtype in [np.double, np.float64]:
            fn_name = 'dump_tb_inputs_double'
            ctype = ctypes.c_double
        else:
            raise Exception(
                'Invalid type ({}) of numpy array. Supported types are: single, float32, double, float64, float_.'.format(
                    first.dtype
                )
            )

        for arr in xlist:
            if arr.dtype != first.dtype:
                raise ValueError('All inputs must have same dtype')
            if not arr.flags['C_CONTIGUOUS']:
                raise ValueError('Input arrays must be C_CONTIGUOUS')

        fn = getattr(self._top_function_lib, fn_name)
        fn.restype = None
        fn.argtypes = [ctypes.c_char_p] + [npc.ndpointer(ctype, flags='C_CONTIGUOUS') for _ in xlist]

        fn(folder_path.encode('ascii'), *xlist)

    def get_input_variables(self):
        variables = []
        for inp in self.inputs:
            variables.append(self.graphs[0].graph[inp].get_output_variable())
        return variables

    def get_layers(self):
        all_values = []
        for g in self.graphs:
            all_values.extend(g.graph.values())
        return dict(zip(all_values, all_values)).values()

    def _get_pragma_details(self, pragma):
        """
        Extracts the pragma type and FIFO depth from the given pragma.
        """
        if isinstance(pragma, str):
            pragma_str = pragma  # 'reshape' or 'partition' pragma
            fifo_depth = 1
        elif isinstance(pragma, (list, tuple)):
            pragma_str = pragma[0]  # 'stream' pragma
            fifo_depth = pragma[1]
        else:
            raise ValueError(f'Unexpected format for pragma: {pragma}')

        return pragma_str, fifo_depth

    def _print_status(self, status):
        print('\r', end='')
        status_icons = {'Pending': '○', 'Running': '⌛', 'Completed': '✅', 'Failed': '❌'}
        status_str = ' | '.join(f'{proj}: {status_icons.get(stat, "?")}' for proj, stat in status.items())
        print(status_str, flush=True)

    def _assert_consistent_pragmas(self):
        """
        Ensure all graphs have the same pragma in their input and output layers.
        Stitching and simulating mixed pragmas is not supported at the moment.
        """
        ref_pragmas = {
            self._get_pragma_details(self.graphs[0].output_vars[layer].pragma)[0]
            for layer in self.graphs[0].inputs + self.graphs[0].outputs
            if layer in self.graphs[0].output_vars
        }

        if len(ref_pragmas) != 1:
            raise ValueError(
                f'Multiple pragmas detected in 1st graph: {ref_pragmas}. '
                'Ensure all graphs have the same interface (stream or partition).'
            )

        for idx, g in enumerate(self.graphs[1:], start=1):
            current_pragmas = {
                self._get_pragma_details(g.output_vars[layer].pragma)[0]
                for layer in g.inputs + g.outputs
                if layer in g.output_vars
            }

            if ref_pragmas != current_pragmas:
                raise ValueError(
                    f'Pragma mismatch in graph {idx}:\n' f'Expected: {ref_pragmas}\n' f'Found: {current_pragmas}'
                )

    def _make_stamp(self):
        length = 8
        stamp = uuid.uuid4()
        return str(stamp)[-length:]

    def _replace_logos(self):
        spec = importlib.util.find_spec('hls4ml')
        hls4ml_path = os.path.dirname(spec.origin)
        hls4ml_logo = os.path.join(hls4ml_path, '../docs/img/logo_small.png')

        if not os.path.isfile(hls4ml_logo):
            raise FileNotFoundError(f'hls4ml logo not found at: {hls4ml_logo}')

        for g in self.graphs:
            graph_logo_paths = [
                os.path.join(
                    g.config.get_output_dir(), g.config.get_project_name() + '_prj', 'solution1/impl/misc/logo.png'
                ),
                os.path.join(
                    g.config.get_output_dir(), g.config.get_project_name() + '_prj', 'solution1/impl/ip/misc/logo.png'
                ),
            ]
            try:
                for logo in graph_logo_paths:
                    shutil.copy(hls4ml_logo, logo)
            except Exception as e:
                print(f'Error copying hls4ml logo to {g.config.get_output_dir()} project: {e}')


def to_multi_model_graph(model: ModelGraph, split_before_layers: list[str]):
    """
    Create a MultiModelGraph by splitting a base ModelGraph before the specified layer names.

    Args:
        model (ModelGraph): the original monolithic model graph
        split_before_layers (list[str]): list of layer names to partition the original model graph.
        Splitting on a not a cut edge (removing that single edge does not split the whole graph) is not allowed.

    Returns:
        multi_model_graph (MultiModelGraph): the partitioned multi model graph
    """
    return MultiModelGraph.from_model_graph(model, split_before_layers)
