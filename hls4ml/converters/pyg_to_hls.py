from __future__ import print_function
from collections import OrderedDict

from hls4ml.converters.pytorch_to_hls import PyTorchModelReader
from hls4ml.model.hls_model import HLSModel
from hls4ml.templates import get_backend

class PygModelReader(PyTorchModelReader):
    def __init__(self, config):
        super().__init__(config)
        self.n_node = config['InputShape']['NodeAttr'][0]
        self.n_edge = config['InputShape']['EdgeAttr'][0]
        self.node_dim = config['InputShape']['NodeAttr'][1]
        self.edge_dim = config['InputShape']['EdgeAttr'][1]

# EdgeBlock/NodeBlock/Aggregate handlers
block_handlers = {}

def register_pyg_block_handler(block_name, handler_func):
    if block_name in block_handlers:
        raise Exception('Block {} already registered'.format(block_name))
    else:
        block_handlers[block_name] = handler_func

def get_supported_pyg_blocks():
    return list(block_handlers.keys())

def pyg_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

def pyg_to_hls(config):
    forward_dict = config['ForwardDictionary']
    activate_final = config['ActivateFinal']

    # get precisions
    backend = get_backend(config.get('Backend', 'Vivado'))
    fp_type = backend.convert_precision_string(config['HLSConfig']['Model']['Precision'])
    int_type = backend.convert_precision_string(config['HLSConfig']['Model']['IndexPrecision'])

    # make reader
    reader = PygModelReader(config)
    n_node = reader.n_node
    n_edge = reader.n_edge
    node_dim = reader.node_dim
    edge_dim = reader.edge_dim

    # initiate layer list with inputs: node_attr, edge_attr, edge_index
    layer_list = []
    input_shapes = reader.input_shape
    NodeAttr_layer = {
        'name': 'node_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['NodeAttr'],
        'inputs': 'input',
        'precision': fp_type
    }
    layer_list.append(NodeAttr_layer)
    EdgeAttr_layer = {
        'name': 'edge_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeAttr'],
        'inputs': 'input',
        'precision': fp_type
    }
    layer_list.append(EdgeAttr_layer)
    EdgeIndex_layer = {
        'name': 'edge_index',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeIndex'],
        'inputs': 'input',
        'precision': int_type
    }
    layer_list.append(EdgeIndex_layer)
    update_dict = {"last_node_update": "node_attr", "last_edge_update": "edge_attr", "last_edge_aggr_update": None}

    # insert an aggregation step before each NodeBlock
    aggr_count = 0
    forward_dict_new = OrderedDict()
    for key, val in forward_dict.items():
        if val == "NodeBlock":
            aggr_count += 1
            aggr_key = f"aggr{aggr_count}"
            aggr_val = "EdgeAggregate"
            forward_dict_new[aggr_key] = aggr_val
        forward_dict_new[key] = val

    # complete the layer list
    for i, (key, val) in enumerate(forward_dict_new.items()):
        # get inputs, outputs
        index = len(layer_list)+1
        layer_dict, update_dict = block_handlers[val](key, config, update_dict, index, n_node, n_edge, node_dim, edge_dim)
        layer_list.append(layer_dict)

    if activate_final is not None:
        act_dict = {
            'name': 'final_act',
            'class_name': 'Activation',
            'inputs': [f"layer{len(layer_list)}_out"],
            'activation': activate_final,
            'precision': fp_type
        }
        layer_list.append(act_dict)
        out = ["final_act"]
    else:
        out = [layer_list[-1]['outputs'][0]]

    hls_model = HLSModel(config, reader, layer_list, inputs=['node_attr', 'edge_attr', 'edge_index'])
    hls_model.outputs = out
    return hls_model

