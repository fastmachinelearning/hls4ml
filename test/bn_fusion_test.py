from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent

if __name__ == "__main__":

    class BatchNormModel(nn.Module):
        def __init__(self, filters, momentum):
            super().__init__()
            self.conv1 = nn.Conv1d(
                int(filters),
                filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm1d(filters, momentum,
                                     track_running_stats=True)
            self.relu1 = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            return x

    n_in = 2 #number of channels
    size_in_height = 1024
    n_batch = 2

    model = BatchNormModel(n_in, 0.2)

    traced_model = symbolic_trace(model)
    traced_model.print_readable()
    for node in traced_model.graph.nodes:
        print("main dbg: " + node.op)

    # channels first.
    X_input = np.random.rand(n_batch, n_in, size_in_height)

    print("X Shape: ", end=" ")
    print(X_input.shape)

    #running inference with model
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()
    print("y Shape: ", end=" ")
    print(pytorch_prediction.shape)
    io_type='io_stream'

    config = config_from_pytorch_model(model,
                                       default_reuse_factor=12,
                                       #granularity='name',
                                       transpose_outputs=False)
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = 'ap_fixed<64,24>'
    print(config)

    backend='Vivado'
    output_dir = str(test_root_path / f'hls4mlprj_block_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model,
        (None, n_in, size_in_height),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
