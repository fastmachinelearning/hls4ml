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
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_merge(test_case_id, merge_op, io_type, backend):
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
        channels_last_conversion='internal',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / test_case_id)
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


# ---------------------------------------------------------------------------- #
# Scalar operands, positional concatenation dimension, and rejects of merges
# hls4ml cannot represent.
# ---------------------------------------------------------------------------- #


class ScalarMergeModel(nn.Module):
    """Applies an elementwise operation with a scalar operand, in either position."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


scalar_merge_ops = {
    'add': lambda x: x + 2.5,
    'sub': lambda x: x - 2.5,
    'rsub': lambda x: 2.5 - x,  # the scalar in the first position used to replace the wrong argument
    'mul': lambda x: x * 0.5,
}


@pytest.mark.parametrize('merge_op', scalar_merge_ops.keys())
def test_merge_scalar_operand(test_case_id, merge_op):
    model = ScalarMergeModel(scalar_merge_ops[merge_op])
    model.eval()

    config = hls4ml.utils.config_from_pytorch_model(model, (8,), default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, io_type='io_parallel', backend='Vivado'
    )
    hls_model.compile()

    X_input = np.round(np.random.rand(10, 8) * 2**10) * 2**-10
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=0.001)


class ConcatPositionalDimModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat([x, y], self.dim)  # the positional dim used to be ignored, defaulting to the batch axis


@pytest.mark.parametrize('dim', [1, -1])
def test_concat_positional_dim(test_case_id, dim):
    model = ConcatPositionalDimModel(dim)
    model.eval()

    config = hls4ml.utils.config_from_pytorch_model(model, [(8,), (8,)], default_precision='ap_fixed<32,16>')
    output_dir = str(test_root_path / test_case_id)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, hls_config=config, output_dir=output_dir, io_type='io_parallel', backend='Vivado'
    )
    hls_model.compile()

    X_input1 = np.round(np.random.rand(10, 8) * 2**10) * 2**-10
    X_input2 = np.round(np.random.rand(10, 8) * 2**10) * 2**-10
    pytorch_prediction = model(torch.Tensor(X_input1), torch.Tensor(X_input2)).detach().numpy()
    hls_prediction = hls_model.predict([X_input1, X_input2])

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=0.001)


def test_concat_batch_dim_rejected(test_case_id):
    class ConcatBatchModel(nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], 0)

    model = ConcatBatchModel()
    model.eval()

    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match='batch dimension'):
        # config_from_pytorch_model traces the model too, so the reject can fire already there
        config = hls4ml.utils.config_from_pytorch_model(model, [(8,), (8,)])
        hls4ml.converters.convert_from_pytorch_model(
            model, hls_config=config, output_dir=output_dir, io_type='io_parallel', backend='Vivado'
        )


def test_merge_broadcast_rejected(test_case_id):
    class BroadcastMergeModel(nn.Module):
        def forward(self, x, y):
            return x + y

    model = BroadcastMergeModel()
    model.eval()

    output_dir = str(test_root_path / test_case_id)
    with pytest.raises(NotImplementedError, match='broadcasting'):
        # config_from_pytorch_model traces the model too, so the reject can fire already there
        config = hls4ml.utils.config_from_pytorch_model(model, [(2, 6), (6,)])
        hls4ml.converters.convert_from_pytorch_model(
            model, hls_config=config, output_dir=output_dir, io_type='io_parallel', backend='Vivado'
        )
