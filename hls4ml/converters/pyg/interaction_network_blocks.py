import numpy as np
from hls4ml.converters.pyg_to_hls import pyg_handler

def parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim):
    layer_dict = {
        "name": block_name,
        "n_node": n_node,
        "n_edge": n_edge,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "activate_final": "false",
        "activation": "linear"
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
def parse_NodeBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim):
    layer_dict = parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim)
    layer_dict["class_name"] = "NodeBlock"
    layer_dict["inputs"] = [update_dict["last_node_update"], update_dict["last_edge_aggr_update"]]#this is where the concat method is described
    layer_dict["outputs"] = [f"layer{index}_out"]
    update_dict["last_node_update"] = f"layer{index}_out"
    return layer_dict, update_dict

@pyg_handler('EdgeBlock')
def parse_EdgeBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim):
    layer_dict = parse_GraphBlock(block_name, config, n_node, n_edge, node_dim, edge_dim)
    layer_dict["class_name"] = "EdgeBlock"
    layer_dict["inputs"] = [update_dict["last_node_update"], update_dict["last_edge_update"], "edge_index"]
    layer_dict["outputs"] = [f"layer{index}_out"]
    update_dict["last_edge_update"] = f"layer{index}_out"
    return layer_dict, update_dict

@pyg_handler('EdgeAggregate')
def parse_EdgeAggregate(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim):
    layer_dict = {"name": f"aggr{index}",
                  "class_name": "EdgeAggregate",
                  "n_node": n_node,
                  "n_edge": n_edge,
                  "node_dim": node_dim,
                  "edge_dim": edge_dim,
                  "out_dim": edge_dim,
                  "inputs": [update_dict["last_edge_update"], "edge_index"],
                  "outputs": [f"layer{index}_out"],
                  "activate_final": "false"}
    update_dict["last_edge_aggr_update"] = f"layer{index}_out"
    return layer_dict, update_dict


@pyg_handler('ResidualBlock')
def parse_ResidualBlock(block_name, config, update_dict, index, n_node, n_edge, node_dim, edge_dim):
    layer_dict = {"name": f"aggr{index}",
                  "class_name": "ResidualBlock",
                  "n_node": n_node,
                  "node_dim": node_dim,
                  "inputs": [update_dict["last_node_update"],update_dict["last_node_update"],],
                  "outputs": [f"layer{index}_out"],
                  "activate_final": "false"}
    update_dict["last_node_update"] = f"layer{index}_out" #bc it comes right after nodeblock
    return layer_dict, update_dict