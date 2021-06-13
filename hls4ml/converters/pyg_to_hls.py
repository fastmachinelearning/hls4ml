from __future__ import print_function
import torch

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

    def get_weights_data(self, layer_name, var_name, module_name=None):
        data = None

        # Parameter mapping from pytorch to keras
        torch_paramap = {
            # Conv
            'kernel': 'weight',
            # Batchnorm
            'gamma': 'weight',
            'beta': 'bias',
            'moving_mean': 'running_mean',
            'moving_variance': 'running_var'}

        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')

        if module_name is not None:
            if var_name in list(torch_paramap.keys()):
                var_name = torch_paramap[var_name]

            try:
                data = self.state_dict[module_name + '.' + layer_name + '.' + var_name].numpy().transpose()
            except KeyError:
                data = self.state_dict[module_name + '.layers.' + layer_name + '.' + var_name].numpy().transpose()

        else:
            if var_name in list(torch_paramap.keys()):
                var_name = torch_paramap[var_name]

            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()  # Look at transpose when systhesis produce lousy results. Might need to remove it.

        return data

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
        'dim_names': ['N_NODE', 'NODE_DIM'],
        'precision': fp_type
    }
    layer_list.append(NodeAttr_layer)
    EdgeAttr_layer = {
        'name': 'edge_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeAttr'],
        'inputs': 'input',
        'dim_names': ['N_EDGE', 'EDGE_DIM'],
        'precision': fp_type
    }
    layer_list.append(EdgeAttr_layer)
    EdgeIndex_layer = {
        'name': 'edge_index',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeIndex'],
        'inputs': 'input',
        'dim_names': ['N_EDGE', 'TWO'],
        'precision': int_type
    }
    layer_list.append(EdgeIndex_layer)
    last_node_update = "node_attr"
    last_edge_update = "edge_attr"

    # If the first block is a NodeBlock, we need a layer to construct the initial edge_aggregates
    if forward_dict[list(forward_dict.keys())[0]] == "NodeBlock":
        aggr_layer = {"name": "aggr1",
                       "class_name": "Aggregate",
                       "n_node": n_node,
                       "n_edge": n_edge,
                       "node_dim": node_dim,
                       "edge_dim": edge_dim,
                       "precision": fp_type,
                       "out_dim": edge_dim,
                       "inputs": ["edge_attr", "edge_index"],
                       "outputs": ["edge_attr_aggr"]}
        layer_list.append(aggr_layer)
        last_edge_aggr_update = "edge_attr_aggr"
    else: last_edge_aggr_update = None

    # complete the layer list
    for i, (key, val) in enumerate(forward_dict.items()):
        layer_dict = {
            "name": key,
            "class_name": val,
            "n_node": n_node,
            "n_edge": n_edge,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "precision": fp_type
        }

        # get n_layers, out_dim
        model = config['PytorchModel']
        torch_block = getattr(model, key)
        try:
            torch_layers = torch_block.layers._modules
        except AttributeError:
            torch_layers = torch_block._modules

        lcount = 0
        for lname, l in torch_layers.items():
            if isinstance(l, torch.nn.modules.linear.Linear):
                lcount += 1
                last_layer = l
        layer_dict["n_layers"] = lcount
        layer_dict["out_dim"] = last_layer.out_features

        # get inputs, outputs
        if val == "NodeBlock":
            index = len(layer_list) + 1
            layer_dict["inputs"] = [last_node_update, last_edge_aggr_update]
            layer_dict["outputs"] = [f"layer{index}_out"]
            last_node_update = f"layer{index}_out"
            layer_list.append(layer_dict)
        elif val == "EdgeBlock":
            index = len(layer_list) + 1
            layer_dict["inputs"] = [last_node_update, last_edge_update, "edge_index"]
            layer_dict["outputs"] = [f"layer{index}_out"]
            last_edge_update = f"layer{index}_out"
            layer_list.append(layer_dict)

        # if val==EdgeBlock and this is not the final graph-block, follow it with an aggregation layer
        if (val == "EdgeBlock") and (i < len(forward_dict) - 1):
            index = len(layer_list) + 1
            layer_dict = {"name": f"aggr{index}",
                       "class_name": "Aggregate",
                       "n_node": n_node,
                       "n_edge": n_edge,
                       "node_dim": node_dim,
                       "edge_dim": edge_dim,
                       "precision": fp_type,
                       "out_dim": edge_dim,
                       "inputs": [last_edge_update, "edge_index"],
                       "outputs": [f"layer{index}_out"]}
            last_edge_aggr_update = f"layer{index}_out"
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

