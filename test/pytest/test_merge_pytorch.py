from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import hls4ml

test_root_path = Path(__file__).parent


class MergeModule(nn.Module):
    def __init__(self, merge_op):
        super().__init__()
        self.op = getattr(torch, merge_op)

    def forward(self, x, y):
        return self.op(x, y)


class ConcatModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # In this test the shape will be (batch, 3, 10, 10), but since we test with channels_last data format, this
        # will be equivalent to the Keras default of concatenation along the last axis (axis=-1)
        return torch.cat([x, y], dim=1)


@pytest.mark.parametrize('merge_op', ['cat', 'add', 'mul', 'sub', 'minimum', 'maximum'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_merge(merge_op, io_type, backend):
    input_shape = (3, 10, 10)

    if merge_op == 'cat':  # Meow!
        model = ConcatModule()
    else:
        model = MergeModule(merge_op)
    model.eval()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        [input_shape, input_shape],
        default_precision='ap_fixed<32,16>',
        channels_last_conversion="internal",
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / f'hls4mlprj_merge_pytorch_{merge_op}_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
    )
    hls_model.compile()

    X_input1 = np.random.rand(100, *input_shape)
    X_input2 = np.random.rand(100, *input_shape)

    X_input1_cl = np.ascontiguousarray(np.transpose(X_input1, axes=[0, 2, 3, 1]))
    X_input2_cl = np.ascontiguousarray(np.transpose(X_input2, axes=[0, 2, 3, 1]))

    pytorch_prediction = model(torch.Tensor(X_input1), torch.Tensor(X_input2)).detach().numpy()
    hls_prediction = hls_model.predict([X_input1_cl, X_input2_cl])

    output_shape = pytorch_prediction.shape
    output_shape_cl = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
    hls_prediction = np.transpose(hls_prediction.reshape(output_shape_cl), axes=[0, 3, 1, 2])

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=0.001)
