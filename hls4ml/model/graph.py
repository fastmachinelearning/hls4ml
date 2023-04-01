import ctypes
import os
import platform
from collections import OrderedDict

import numpy as np
import numpy.ctypeslib as npc

from hls4ml.backends import get_backend
from hls4ml.model.flow import get_flow
from hls4ml.model.layers import layer_map
from hls4ml.model.optimizer import get_available_passes, optimize_model


class HLSConfig:
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

        self.model_strategy = 'Latency'
        self.layer_type_strategy = {}
        self.layer_name_strategy = {}

        self.model_conv_implementation = 'LineBuffer'
        self.layer_type_conv_implementation = {}
        self.layer_name_conv_implementation = {}

        self.model_compression = False
        self.layer_type_compression = {}
        self.layer_name_compression = {}

        self.trace_output = self.get_config_value('TraceOutput', False)

        self._parse_hls_config()
        self._validate_hls_config()

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def get_project_name(self):
        return self.get_config_value('ProjectName')

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

    def get_precision(self, layer, var='default'):
        precision = self.layer_name_precision.get(layer.name.lower() + '_' + var)
        type_name = layer.name.lower() + '_' + var + '_t'
        if precision is None:
            precision = self.layer_name_precision.get(layer.name.lower() + '_default')
            type_name = layer.name.lower() + '_default_t'

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
            self.model_strategy = model_cfg.get('Strategy', 'Latency')
            self.model_compression = bool(model_cfg.get('Compression', 0))

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
                    self.layer_type_strategy[layer_type.lower()] = strategy

                conv_implementation = layer_cfg.get('ConvImplementation')
                if conv_implementation is not None:
                    self.layer_type_conv_implementation[layer_type.lower()] = conv_implementation

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_type_compression[layer_type.lower()] = bool(compression)

        layer_name_cfg = hls_config.get('LayerName')
        if layer_name_cfg is not None:
            for layer_name, layer_cfg in layer_name_cfg.items():
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
                    self.layer_name_strategy[layer_name.lower()] = strategy

                conv_implementation = layer_cfg.get('ConvImplementation')
                if conv_implementation is not None:
                    self.layer_name_conv_implementation[layer_name.lower()] = conv_implementation

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_name_compression[layer_name.lower()] = bool(compression)

    def _validate_hls_config(self):
        use_resource = False
        if self.model_strategy.lower() == 'latency' and self.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            use_resource = True
        for layer_type, strategy in self.layer_type_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print(
                    'WARNING: Strategy for layer type {} set to "Resource", while model strategy set to "Latency".'.format(
                        layer_type
                    )
                )
                use_resource = True

        for layer_name, strategy in self.layer_name_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print(
                    'WARNING: Strategy for layer {} set to "Resource", while model strategy set to "Latency".'.format(
                        layer_name
                    )
                )
                use_resource = True

        for layer_type, compression in self.layer_type_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print(
                    'WARNING: Compression enabled for layer type {}, while model strategy set to "Latency".'.format(
                        layer_type
                    )
                )
                use_resource = True

        for layer_name, compression in self.layer_name_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print(f'WARNING: Compression enabled for layer {layer_name}, while model strategy set to "Latency".')
                use_resource = True

        if use_resource:
            print('WARNING: Changing model strategy to "Resource"')
            self.model_strategy = 'Resource'


class ModelGraph:
    """The ModelGraph represents the network that is being processed by hls4ml.

    Args:
        config (dict):  The configuration dictionary
        data_reader:  The data reader from where weights can be extracted
        layer_list (list(dict)):  The list contains a dictionary for each input layer
        inputs (list, optional):  The inputs to the model. If None, determined from layer_list
        outputs (list, optional):  The outputs to the model. If None, determined from layer_list
    """

    def __init__(self, config, data_reader, layer_list, inputs=None, outputs=None):
        self.config = HLSConfig(config)
        self.reader = data_reader

        # keep track of the applied flows
        self._applied_flows = []

        # If not provided, assumes layer_list[0] is the input layer, and layer_list[-1] is output layer

        # Note, these are actually the variable names, which may differ from the layer name
        input_layers = inputs if inputs is not None else [layer_list[0]['name']]
        output_layers = outputs if outputs is not None else [layer_list[-1]['name']]
        self.inputs = self._find_output_variable_names(layer_list, input_layers)
        if self.inputs != input_layers:
            raise RuntimeError(
                "Currently only support the case when input variables and input layer names match\n"
                + f"Input layers = {input_layers}, input_vars = {self.inputs}"
            )
        self.outputs = self._find_output_variable_names(layer_list, output_layers)

        self.index = 0
        self.graph = OrderedDict()  # where the nodes are stored
        self.output_vars = {}

        self._top_function_lib = None

        self._make_graph(layer_list)

        for flow in self.config.flows:
            self.apply_flow(flow)

    def _find_output_variable_names(self, layer_list, layer_names):
        """Given a list of all layers, and a list input/output names, find the names of the their outputs that will be used
        as the name of the output variables."""
        inout_nodes = [node for node in layer_list if node['name'] in layer_names]
        all_node_output_names = [node['outputs'] if 'outputs' in node else [node['name']] for node in inout_nodes]
        return [output for node_output_names in all_node_output_names for output in node_output_names]  # to flatten

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

    def make_node(self, kind, name, attributes, inputs, outputs=None):
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
        node = layer_cls(self, name, attributes, inputs, outputs)
        for o in node.outputs:
            out_var = node.get_output_variable(output_name=o)
            if o in self.outputs:
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
            next_node = next((x for x in self.graph.values() if x.inputs[0] in prev_node.outputs), None)
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

        new_graph = OrderedDict()
        for k, v in self.graph.items():
            new_graph[k] = v
            if k == prev_node.name:
                new_graph[node.name] = node

        self.graph = new_graph
        self._update_model_outputs()

    def remove_node(self, node, rewire=True):
        """Remove a node from a graph.

        By default, this function can connect the outputs of previous node to the input of next one.
        Note that when removing a leaf node `rewire` should be set to `False`.

        Args:
            node (Layer): The node to remove
            rewire (bool, optional): If `True`, connects the outputs of the previous node
                to the inputs of the next node

        Raises:
            Exception: If an attempt is made to rewire a leaf node or a node with multiple
                inputs/outputs.

        """
        if rewire:
            inputs = [inp for inp in node.inputs if inp]
            outputs = [outp for outp in node.outputs if outp]
            if len(inputs) > 1 or len(outputs) > 1:
                raise Exception('Cannot rewire a node with multiple inputs/outputs')
            prev_node = node.get_input_node(node.inputs[0])
            next_nodes = [x for x in self.graph.values() if node.outputs[0] in x.inputs]
            if prev_node is not None:
                if len(next_nodes) > 0:
                    for next_node in next_nodes:
                        for i, _ in enumerate(next_node.inputs):
                            if node.outputs[0] == next_node.inputs[i]:
                                next_node.inputs[i] = prev_node.outputs[0]
                                break
                else:
                    if not node.outputs[0] in self.outputs:
                        raise Exception('Cannot rewire a node without child')
            else:
                raise Exception('Cannot rewire a node without a parent')

        del self.output_vars[node.outputs[0]]
        del self.graph[node.name]
        self._update_model_outputs()

    def replace_node(self, old_node, new_node):
        """Replace an existing node in the graph with a new one.

        Args:
            old_node (Layer): The node to replace
            new_node (Layer): The new node

        """
        prev_node = self.graph.get(old_node.inputs[0])
        next_node = next((x for x in self.graph.values() if x.inputs[0] == old_node.outputs[0]), None)
        if next_node is not None:
            next_node.inputs[0] = new_node.outputs[0]
        if prev_node is not None:
            if new_node.inputs is None or len(new_node.inputs) == 0:  # Check if already rewired
                new_node.inputs = [prev_node.outputs[0]]

        self.graph = OrderedDict((new_node.name, new_node) if k == old_node.name else (k, v) for k, v in self.graph.items())
        self._update_model_outputs()

    def _update_model_outputs(self):
        '''Update the model outputs

        All node outputs and inputs are found. The model outputs are set to all node outputs
        that are not also node inputs.
        '''
        node_outputs = [out for node in self.graph.values() for out in node.outputs]
        node_inputs = [inp for node in self.graph.values() for inp in node.inputs]
        self.outputs = [out for out in node_outputs if out not in node_inputs]

    def get_weights_data(self, layer_name, var_name):
        return self.reader.get_weights_data(layer_name, var_name)

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
        if out_name in self.outputs:
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
        elif x0.dtype in [np.double, np.float64, np.float_]:
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

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir() + '/firmware')

        output = []
        if n_samples == 1 and n_inputs == 1:
            x = [x]

        try:
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

            # Convert to list of numpy arrays (one for each output)
            output = [
                np.asarray([output[i_sample][i_output] for i_sample in range(n_samples)]) for i_output in range(n_outputs)
            ]
        finally:
            os.chdir(curr_dir)

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
