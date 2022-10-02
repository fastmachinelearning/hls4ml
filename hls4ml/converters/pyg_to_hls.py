from __future__ import print_function
from collections import OrderedDict
import re
import copy

from hls4ml.converters.pytorch_to_hls import PyTorchModelReader
from hls4ml.model.hls_model import HLSModel
from hls4ml.templates import get_backend

class PygModelReader(PyTorchModelReader):
    def __init__(self, config):
        super().__init__(config)
        self.n_node = config['InputShape']['NodeAttr'][0]
        self.n_edge = config['InputShape']['EdgeAttr'][0]
        # self.node_dim = config['InputShape']['NodeAttr'][1]
        # self.edge_dim = config['InputShape']['EdgeAttr'][1]
        # self.common_dim = config['InputShape']['CommonDim']
        self.node_attr = config['InputShape']['NodeAttr'][1]
        self.edge_attr = config['InputShape']['EdgeAttr'][1]
        
        self.node_dim = config['InputShape']['NodeDim']
        self.edge_dim = config['InputShape']['EdgeDim']
        # it should be node_dim == edge_dim
        assert(self.node_dim == self.edge_dim )

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

            # print(f"self.state_dict.keys(): {self.state_dict.keys()}")
            # print(f"layer_name + '.' + var_name: {layer_name + '.' + var_name}")
            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()  # Look at transpose when systhesis produce lousy results. Might need to remove it.

        return data

# EdgeBlock/NodeBlock/Aggregate handlers
block_handlers = {}

def register_pyg_block_handler(block_name, handler_func):
    if block_name in block_handlers:
        raise Exception('Block {} already registered'.format(block_name))
    else:
        block_handlers[block_name] = handler_func

def get_supported_pyg_blocks():
    return list(block_handlers.keys())

def pyg_handler(*args): #used by block_handlers
    def decorator(function):
        function.handles = [arg for arg in args]
        print(f"handler args: {args}")
        return function
    return decorator

##########################################dataflow optimization#########################################
#spaghetti code, please be patient

# helper functions
def get_layer(layer_list, layer_name):
    for idx, layer in enumerate(layer_list):
        if layer["name"]==layer_name:
            return idx, layer

def get_vars_to_clone(layer_list):
    """
    layer_list: list, of the form returned by pyg_to_hls(), pytorch_to_hls(), etc.

    Returns 'vars_to_clone', a nested-dictionary whose
    keys are the names of each variable in 'input_vars'
    which is an input to several layers in 'layer_list'

    'vars_to_clone' format:
    {var0:
        {'creator' (str): layer whose output is var1,
        'receivers' (list of str): list of layers whose inputs are var1,
        'precision' (hls4ml fixed/integer precision object): optional; if the
        original var has this parameter then it is copied over to the clone,
        'pragma' (tuple or str): optional; if the original var has this parameter
        then it is copied over to the clone},
    var1: ...
    }
    """
    optional_params = ["precision", "pragma"]
    input_vars = []
    for layer in layer_list:
        if layer["inputs"] != "input":
            input_vars.extend(layer["inputs"])
    input_vars = list(set(input_vars))

    vars_to_clone = {}
    for var in input_vars:

        var_creators = []
        var_receivers = []
        for layer in layer_list:

            # get var_creator
            try:
                layer_outputs = layer["outputs"]
            except KeyError:
                layer_outputs = [layer["name"]]
            if var in layer_outputs:
                var_creators.append(layer["name"])
            # get var_receivers
            if var in layer["inputs"]:
                var_receivers.append(layer["name"])

        #make sure each var is the output of exactly one layer
        assert(len(var_creators))==1
        # we only care about vars which are inputs to several layers
        if len(var_receivers)>1:
            vars_to_clone[var] = {"creator": var_creators[0], "receivers": var_receivers}

            creator_idx, creator_layer = get_layer(layer_list, var_creators[0])
            for p in optional_params:
                if p in creator_layer:
                    vars_to_clone[var][p] = creator_layer[p]

    return vars_to_clone

def make_clones(vars_to_clone):
    """
    vars_to_clone: dict, of the form returned by get_vars_to_clone()

    Returns 'clone_dict', a nested-dictionary with the same keys as vars_to_clone

    'clone_dict' format:
    {var0:
      {'creator' (str): name of layer whose output is var1,
        'receivers' (list of str): list of layer (names) whose inputs are var1,
        'precision' (hls4ml fixed/integer precision object): optional; if the
        original var has this parameter then it is copied over to the clone,
        'pragma' (tuple or str): optional; if the original var has this parameter
        then it is copied over to the clone,
        'clone_layers' (list of dict): list of clone-layers
        'clone_vars' (list of str): list of clone-variable names
      },
    var1: ...
    }
    """
    optional_params = ["precision", "pragma"]
    clone_dict = {}

    for var, var_dict in vars_to_clone.items():
        clone_dict[var] = var_dict
        clone_dict[var]["clone_layers"] = []
        clone_dict[var]["clone_vars"] = []

        n_clones_required = len(var_dict["receivers"]) - 1
        n_clones_created = 0
        n_usable_clones = 0

        # initialize the first clone layer; necessary because further clones cannot be cloned from the original var
        clone_layer_0 = {"name": f"{var}_clone_{n_usable_clones}",
                         "class_name": "CloneParallel",
                         "outputs": [f"{var}_cpy{n_clones_created + 1}", f"{var}_cpy{n_clones_created + 2}"],
                         "inputs": [var],
                         "out_names": [f"{var}_cpy{n_clones_created + 1}", f"{var}_cpy{n_clones_created + 2}"]
                         }
        for p in optional_params:
            if p in var_dict:
                clone_layer_0[p] = var_dict[p]
        clone_dict[var]["clone_layers"].append(clone_layer_0)
        n_clones_created += 2
        n_usable_clones += 1

        while n_usable_clones < n_clones_required:
            clone_layer_i = {
                "name": f"{var}_clone_{n_usable_clones}",
                "class_name": "CloneParallel",
                "outputs": [f"{var}_cpy{n_clones_created + 1}", f"{var}_cpy{n_clones_created + 2}"],
                "inputs": [clone_dict[var]["clone_layers"][-1]["outputs"][0]],
                "out_names": [f"{var}_cpy{n_clones_created + 1}", f"{var}_cpy{n_clones_created + 2}"]
            }
            for p in optional_params:
                if p in var_dict:
                    clone_layer_i[p] = var_dict[p]
            clone_dict[var]["clone_layers"].append(clone_layer_i)
            n_clones_created += 2
            n_usable_clones += 1

        for idx, layer in enumerate(clone_dict[var]["clone_layers"]):
            if idx==len(clone_dict[var]["clone_layers"])-1:
                clone_dict[var]["clone_vars"].extend(layer["outputs"])
            else:
                clone_dict[var]["clone_vars"].append(layer["outputs"][1])


    return clone_dict

def fix_indeces(layer_list):

    # create layerX_map: maps old layer-indeces to new layer-indeces, if the two aren't the same
    layer_list_old = [i for i in layer_list if i["class_name"]!="CloneParallel"]
    layerX_map = {}
    for idx_old_min_1, layer_old in enumerate(layer_list_old):
        idx_old = idx_old_min_1+1
        layerX_old = f"layer{idx_old}"
        for idx_new_min_1, layer in enumerate(layer_list):
            if layer["name"]==layer_old["name"]:
                idx_new = idx_new_min_1+1

                if idx_new==idx_old:
                    continue
                layerX_new = f"layer{idx_new}"
                layerX_map[layerX_old] = layerX_new

    # replace all instances of old layer-indeces with new layer-indeces
    for layer in layer_list:

        # handle 'inputs'
        if isinstance(layer["inputs"], list):
            for idx, var in enumerate(layer["inputs"]):
                for layerX in layerX_map:
                    if re.search(layerX, var):
                        var = var.replace(layerX, layerX_map[layerX])
                        layer["inputs"][idx] = var
                        break
        else:
            for layerX in layerX_map:
                if re.search(layerX, layer["inputs"]):
                    layer["inputs"] = layer["inputs"].replace(layerX, layerX_map[layerX])
                    break

        # handle 'outputs'
        if "outputs" in layer:
            for idx, var in enumerate(layer["outputs"]):
                for layerX in layerX_map:
                    if re.search(layerX, var):
                        var = var.replace(layerX, layerX_map[layerX])
                        layer["outputs"][idx] = var
                        break

        # handle 'out_names'
        if "out_names" in layer:
            for idx, var in enumerate(layer["out_names"]):
                for layerX in layerX_map:
                    if re.search(layerX, var):
                        var = var.replace(layerX, layerX_map[layerX])
                        layer["out_names"][idx] = var
                        break

        # handle 'name'
        for layerX in layerX_map:
            if re.search(layerX, layer["name"]):
                layer["name"] = layer["name"].replace(layerX, layerX_map[layerX])
                break

        # handle 'pragma' factor
        for layerX in layerX_map:
            old_factor = layer["pragma"][2]
            if re.search(layerX, old_factor):
                new_factor = old_factor.replace(layerX, layerX_map[layerX])
                layer["pragma"] = ("partition", "cyclic", new_factor, 1)
                break

# main function
def optimize_dataflow(layer_list, activate_final):
    layer_list = copy.deepcopy(layer_list)

    # 1. handle pragmas
    for idx, l in enumerate(layer_list):
        try:
            n_cols = l["dim_names"][1].lower()
        except KeyError:
            n_cols = f"layer{idx + 1}_out_dim"
        l["pragma"] = ("partition", "cyclic", n_cols, 1)

    # 2. handle final activation
    if activate_final is not None:
        layer_list[-1]["activate_final"] = "true"
        layer_list[-1]["activation"] = activate_final

    # 3. get all variables which require clones (i.e. they are inputs to 2 or more layers)
    vars_to_clone = get_vars_to_clone(layer_list)

    # 4. make the clones
    clone_dict = make_clones(vars_to_clone)

    # 5. insert the clone layers into the new layer list
    new_layer_list = []
    for layer in layer_list:
        new_layer_list.append(layer)
        for var, var_dict in clone_dict.items():
            if layer["name"] == var_dict["creator"]:
                clone_layers = var_dict["clone_layers"]
                new_layer_list.extend(clone_layers)

    # 6. change the layer inputs to clones wherever necessary
    input_map = {}
    for var, var_dict in clone_dict.items():
        for i, lname in enumerate(var_dict["receivers"]):
            if lname not in input_map.keys():
                input_map[lname] = {}
            input_map[lname][var] = var_dict["clone_vars"][i]

    for layer in new_layer_list:
        if layer["name"] in input_map.keys():
            for inp_idx, inp in enumerate(layer["inputs"]):
                layer["inputs"][inp_idx] = input_map[layer["name"]].get(inp, inp)

    # inserting layers messes up all the "layer{index}" configurations, so this fixes them
    fix_indeces(new_layer_list)
    return new_layer_list

##########################################pyg_to_hls#####################################################

def pyg_to_hls(config):
    forward_dict = config['ForwardDictionary']
    activate_final = config['ActivateFinal']

    # get precisions
    backend = get_backend(config.get('Backend', 'Vivado'))  #returns backend object
    fp_type = backend.convert_precision_string(config['HLSConfig']['Model']['Precision'])
    int_type = backend.convert_precision_string(config['HLSConfig']['Model']['IndexPrecision'])

    # make reader
    
    reader = PygModelReader(config)
    n_node = reader.n_node
    n_edge = reader.n_edge
    node_dim = reader.node_dim
    edge_dim = reader.edge_dim
    node_attr = reader.node_attr
    edge_attr = reader.edge_attr
    print(f"PygModelReader node_dim: {node_dim}")
    print(f"PygModelReader edge_dim: {edge_dim}")
    print(f"PygModelReader node_attr: {node_attr}")
    print(f"PygModelReader edge_attr: {edge_attr}")

    # initiate layer list with inputs: node_attr, edge_attr, edge_index
    layer_list = [] # the order of this list doesn't matter
    input_shapes = reader.input_shape

    NodeAttr_layer = {
        'name': 'node_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['NodeAttr'],
        'inputs': 'input',
        'dim_names': ['N_NODE', 'NODE_DIM'], # here, NODE_DIM == node_attr. possible point of confusion
        'precision': fp_type
    }
    layer_list.append(NodeAttr_layer)

    EdgeAttr_layer = {
        'name': 'edge_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeAttr'],
        'inputs': 'input',
        'dim_names': ['N_EDGE', 'EDGE_DIM'],# here, EDGE_DIM == edge_attr. possible point of confusion
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

    update_dict = {"last_node_update": "node_attr", "last_edge_update": "edge_attr", "last_edge_aggr_update": None}

    # insert an aggregation step before each NodeBlock
    aggr_count = 0
    forward_dict_new = OrderedDict()
    for key, val in forward_dict.items():
        if val == "NodeBlock":
            # add EdgeAggregate b4 NodeBlock in forward dict
            aggr_count += 1
            aggr_key = f"aggr{aggr_count}"
            # just to give a distinction as aggregation blocks are not
            # named variables in the pyg model
            aggr_val = "EdgeAggregate"
            # my guess is that EdgeAggregate block is equivalent to 
            # aggregate() portion of pyg model
            forward_dict_new[aggr_key] = aggr_val
        forward_dict_new[key] = val

    # complete the layer list
    for i, (key, val) in enumerate(forward_dict_new.items()):
        # get inputs, outputs
        index = len(layer_list)+1
        # print(f"block_handlers.keys(): {block_handlers.keys()}")
        layer_dict, update_dict = block_handlers[val](key, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr)
        # possible block hander is [parse_NodeBlock, parse_EdgeBlock, parse_EdgeAggregate]
        print(f"{key} layer_dict: {layer_dict}")
        layer_list.append(layer_dict)

    # handle dataflow optimization
    if config["gnn_resource_limit"] == "true":
        layer_list = optimize_dataflow(layer_list, activate_final)
        out = [layer_list[-1]['outputs'][0]]
    else:
        # handle final activation
        if activate_final is not None:
            activation_dict = {
                'name': 'final_act',
                'class_name': 'Activation',
                'inputs': [f"layer{len(layer_list)}_out"],
                'activation': activate_final,
                'precision': fp_type
            }
            layer_list.append(activation_dict)
            out = ["final_act"]
        else:
            out = [layer_list[-1]['outputs'][0]]

    hls_model = HLSModel(config, reader, layer_list, inputs=['node_attr', 'edge_attr', 'edge_index'])
    hls_model.outputs = out
    return hls_model

