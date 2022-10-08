import numpy as np
from hls4ml.converters.pyg_to_hls import pyg_handler

def parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    """
    graphblocks don't need node_attr and edge_attr
    """
    layer_dict = {
        "name": block_name,
        "n_node": n_node,
        "n_edge": n_edge,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "activate_final": "false",
        "activation": "linear",
    }

    # get n_layers, out_dim
    model = config['PytorchModel']
    torch_block = getattr(model, block_name)
    try:
        torch_layers = torch_block.layers._modules
    except AttributeError:
        torch_layers = torch_block._modules

    lcount = 0
    for lname, l in torch_layers.items():
        if l.__class__.__name__=="Linear":
            lcount += 1
            last_layer = l
    layer_dict["n_layers"] = lcount
    layer_dict["out_dim"] = last_layer.out_features
    return layer_dict

@pyg_handler('NodeBlock')
def parse_NodeBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr)
    layer_dict["class_name"] = "NodeBlock"
    layer_dict["inputs"] = [update_dict["last_node_update"], update_dict["last_edge_aggr_update"]]
    layer_dict["outputs"] = [f"layer{index}_out"]
    layer_dict["n_edge"] = n_edge
    # print(f"type(update_dict): {type(update_dict)}")
    update_dict["last_last_node_update"] = update_dict["last_node_update"] #for residual block
    update_dict["last_node_update"] = f"layer{index}_out"
    # print(f"nodeblock layer_dict: {layer_dict}")
    return layer_dict, update_dict

# @pyg_handler('EdgeBlock')
# def parse_EdgeBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
#     layer_dict = parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr)
#     layer_dict["class_name"] = "EdgeBlock"
#     layer_dict["inputs"] = [update_dict["last_node_update"], update_dict["last_edge_update"], "edge_index"]
#     layer_dict["outputs"] = [f"layer{index}_out"]
#     # print(f'EdgeBlock update_dict["last_edge_update"] b4: {update_dict["last_edge_update"] }')
#     update_dict["last_edge_update"] = f"layer{index}_out"
#     # print(f'EdgeBlock update_dict["last_edge_update"] after: {update_dict["last_edge_update"] }')
#     return layer_dict, update_dict

@pyg_handler('EdgeAggregate')
def parse_EdgeAggregate(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr, Beta):
    layer_dict = {"name": f"aggr{index}",
                  "class_name": "EdgeAggregate",
                  "n_node": n_node,
                  "n_edge": n_edge,
                  "node_dim": node_dim,
                  "edge_dim": edge_dim,
                  "out_dim": edge_dim,
                  "inputs": [update_dict["last_node_update"], update_dict["last_edge_update"], "edge_index"],
                  "outputs": [f"layer{index}_out"],
                  "activate_final": "false",
                  "Beta" : Beta}
    update_dict["last_edge_aggr_update"] = f"layer{index}_out"
    # print(f"aggregate n_edge: {n_edge}")
    # print(f"aggregate last_node_update: {update_dict['last_node_update']}, last_edge_update: {update_dict['last_edge_update']}")
    return layer_dict, update_dict


@pyg_handler('ResidualBlock')
def parse_ResidualBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = {
                "name": f"aggr{index}",
                "class_name": "ResidualBlock",
                "n_node": n_node,
                "node_dim": node_dim,
                "inputs": [update_dict["last_last_node_update"],update_dict["last_node_update"],],
                "outputs": [f"layer{index}_out"],
                "activate_final": "false"}
    # print(f"layer_dict['inputs']: {layer_dict['inputs']}")
    update_dict["last_node_update"] = f"layer{index}_out" #bc it comes right after nodeblock
    return layer_dict, update_dict


@pyg_handler('NodeEncoder')
def parse_NodeEncoder(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = {
                "name": f"node_encoder",
                "class_name": "NodeEncoder", #"Dense", 
                "n_in": node_attr,
                "n_out": node_dim,
                "n_rows" : n_node,
                "n_cols" : node_attr,
                "inputs": [update_dict["last_node_update"]],
                "outputs": [f"layer{index}_out"],
                "activate_final": "false"}
    update_dict["last_node_update"] = f"layer{index}_out" 
    # print(f"node encoder layer_dict['n_in']: {layer_dict['n_in']}")
    # print(f"node encoder layer_dict['n_out']: {layer_dict['n_out']}")
    
    return layer_dict, update_dict

@pyg_handler('EdgeEncoder')
def parse_EdgeEncoder(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = {
                "name": f"edge_encoder",
                "class_name": "EdgeEncoder", #"Dense",
                "n_in": edge_attr,
                "n_out": edge_dim,
                "n_rows" : n_edge,
                "n_cols" : edge_attr,
                "inputs": [update_dict["last_edge_update"]],
                "outputs": [f"layer{index}_out"],
                "activate_final": "false"}
    # print(f'edge encoder b4 update_dict["last_edge_update"]: {update_dict["last_edge_update"]}')
    update_dict["last_edge_update"] = f"layer{index}_out" 
    # print(f'edge encoder after update_dict["last_edge_update"]: {update_dict["last_edge_update"]}')
    return layer_dict, update_dict

@pyg_handler('NodeEncoderBatchNorm1d')
def parse_NodeEncoderBatchNorm1d(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = {
                "name": f"node_encoder_norm",
                "class_name": "BatchNorm2D", 
                "n_rows" : n_node,
                "n_in": node_dim, #interchangeable with edge_dim
                "n_filt" : -1,
                "inputs": [update_dict["last_node_update"]],
                "outputs": [f"layer{index}_out"]}
    update_dict["last_node_update"] = f"layer{index}_out" 
    # print(f"NodeEncoderBatchNorm1d node_dim: {node_dim}")
    return layer_dict, update_dict

@pyg_handler('EdgeEncoderBatchNorm1d')
def parse_EdgeEncoderBatchNorm1d(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim, node_attr, edge_attr):
    layer_dict = {
                "name": f"edge_encoder_norm",
                "class_name": "BatchNorm2D", 
                "n_rows" : n_edge,
                "n_in": edge_dim, #interchangeable with node_dim
                "n_filt" : -1,
                "inputs": [update_dict["last_edge_update"]],
                "outputs": [f"layer{index}_out"]}
    update_dict["last_edge_update"] = f"layer{index}_out" 
    # print(f"EdgeEncoderBatchNorm1d node_dim: {edge_dim}")
    return layer_dict, update_dict

"""
note: 
-parse nodeblock, edgeblock and edgaggregates don't need node_dim nor edge_dims anymore

-I tried class_name of Node and Edge Encoders to be "Dense", so that 
I could directly call dense layers, but then I would get errors like
'aggr4.weight' for layer_name + '.' + var_name

"""