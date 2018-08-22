import numpy as np
import torch
import torch.nn as nn

from model import SegmentClassifier
from estimator import Estimator
from graph import load_graphs, SparseGraph, graph_from_sparse

import os
import sys
filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "../../", "hls-writer"))
from hls_writer import print_array_to_cpp

feature_scale = np.array([1000., np.pi/8, 1000.])
n_features = feature_scale.shape[0]

cuda = False

if cuda:
    np_to_torch = lambda x, volatile=False, dtype=np.float32: (torch.tensor(x.astype(dtype), requires_grad=False).cuda())
else:
    np_to_torch = lambda x, volatile=False, dtype=np.float32: (torch.tensor(x.astype(dtype), requires_grad=False))

if cuda:
    torch_to_np = lambda x: x.cpu().numpy()
else:
    torch_to_np = lambda x: x.detach().numpy()

hidden_dim = 4
n_iters = 1
model = SegmentClassifier(input_dim=n_features, hidden_dim=hidden_dim, n_iters=n_iters)
estim = Estimator(model, loss_func=nn.BCELoss(), cuda=cuda, l1= 0)
#estim.load_checkpoint('model_3_1iteration.pt')
estim.load_checkpoint('model_1iteration.pt')

#graph = load_graphs(['graph000001_3.npz'], SparseGraph)
graph = load_graphs(['graph000001.npz'], SparseGraph)

g = graph[0]
g = graph_from_sparse(g)

X = np_to_torch(g.X, volatile=False)
Ri = np_to_torch(g.Ri, volatile=False) 
Ro = np_to_torch(g.Ro, volatile=False)

print("X", X)
print("Ri", Ri)
print("Ro", Ro)

print_array_to_cpp("wX",g.X, './')
print_array_to_cpp("wRi",g.Ri, './')
print_array_to_cpp("wRo",g.Ro, './')

print("w1", model.input_network[0].weight.data.transpose(0,1))
print("b1", model.input_network[0].bias.data)

print_array_to_cpp("w1",torch_to_np(model.input_network[0].weight.transpose(0,1)), './')
print_array_to_cpp("b1",torch_to_np(model.input_network[0].bias), './')

X = X.view(1, X.shape[0], X.shape[1])
Ri = Ri.view(1, Ri.shape[0], Ri.shape[1])
Ro = Ro.view(1, Ro.shape[0], Ro.shape[1])

H_logits = model.input_network[0](X)
print ("H_logits", H_logits)

H = model.input_network(X)
print ("H", H)

print("w2", model.edge_network.network[0].weight.data.transpose(0,1))
print("b2", model.edge_network.network[0].bias.data)

print_array_to_cpp("w2",torch_to_np(model.edge_network.network[0].weight.data.transpose(0,1)), './')
print_array_to_cpp("b2",torch_to_np(model.edge_network.network[0].bias), './')

print("w3", model.edge_network.network[2].weight.data.transpose(0,1))
print("b3", model.edge_network.network[2].bias.data)

print_array_to_cpp("w3",torch_to_np(model.edge_network.network[2].weight.data.transpose(0,1)), './')
print_array_to_cpp("b3",torch_to_np(model.edge_network.network[2].bias), './')

H = torch.cat([H, X], dim=-1)
e_temp = model.edge_network(H, Ri, Ro) 
print("e_temp", e_temp)

print("w4", model.node_network.network[0].weight.data.transpose(0,1))
print("b4", model.node_network.network[0].bias.data)

print_array_to_cpp("w4",torch_to_np(model.node_network.network[0].weight.data.transpose(0,1)), './')
print_array_to_cpp("b4",torch_to_np(model.node_network.network[0].bias), './')

print("w5", model.node_network.network[2].weight.data.transpose(0,1))
print("b5", model.node_network.network[2].bias.data)

print_array_to_cpp("w5",torch_to_np(model.node_network.network[2].weight.data.transpose(0,1)), './')
print_array_to_cpp("b5",torch_to_np(model.node_network.network[2].bias), './')

H = model.node_network(H, e_temp, Ri, Ro)
print("H", H)

e = model.forward([X, Ri, Ro]) 
print("e", e)
