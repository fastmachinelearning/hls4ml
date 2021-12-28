import numpy as np
from contrib.interaction_network import InteractionNetwork
import hls4ml
from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model
from collections import OrderedDict
import pytest
import torch

flow = 'source_to_target'
aggr = 'add'
hidden_size = 8
reuse_factor = 8
n_node = 28
n_edge = 56
node_dim = 3
edge_dim = 4

@pytest.fixture(scope='module')
def interaction_network_models():

    model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=hidden_size)

    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    graph_dims = {
        "n_node": n_node,
        "n_edge": n_edge,
        "node_dim": node_dim,
        "edge_dim": edge_dim
    }

    output_dir = 'hls4mlprj_interaction_network'
    config = config_from_pyg_model(model,
                                   default_precision='ap_fixed<16,6>',
                                   default_index_precision='ap_uint<8>',
                                   default_reuse_factor=reuse_factor)
    hls_model = convert_from_pyg_model(model,
                                       forward_dictionary=forward_dict,
                                       activate_final='sigmoid',
                                       output_dir=output_dir,
                                       hls_config=config,
                                       **graph_dims)
    hls_model.compile()
    return model, hls_model


def test_accuracy(interaction_network_models):
    model, hls_model = interaction_network_models
    node_attr = (torch.rand(n_node, node_dim)-0.5)*2**6
    edge_attr = (torch.rand(n_edge, edge_dim)-0.5)*2**6
    edge_index = torch.randint(n_node, size=(2, n_edge))

    y = model(node_attr,
              edge_index,
              edge_attr).detach().cpu().numpy()

    node_attr, edge_attr, edge_index = node_attr.detach().cpu().numpy(), edge_attr.detach().cpu().numpy(), edge_index.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    node_attr, edge_attr, edge_index = np.ascontiguousarray(node_attr), np.ascontiguousarray(edge_attr), np.ascontiguousarray(edge_index)
    x_hls = [node_attr, edge_attr, edge_index]
    y_hls = hls_model.predict(x_hls).reshape(y.shape)

    np.testing.assert_allclose(y_hls, y, rtol=0, atol=0.02)
