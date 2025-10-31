from pathlib import Path

import numpy as np
import torch.nn as nn
from onnx import TensorProto, helper

from hls4ml.converters import convert_from_onnx_model, convert_from_pytorch_model
from hls4ml.utils.config import config_from_onnx_model, config_from_pytorch_model

test_root_path = Path(__file__).parent


def _make_constantpad_onnx_1d():
    input_tensor = helper.make_tensor_value_info('global_in', TensorProto.FLOAT, [1, 2, 4])
    output_tensor = helper.make_tensor_value_info('global_out', TensorProto.FLOAT, [1, 2, 9])
    pads_tensor = helper.make_tensor_value_info('pads', TensorProto.INT64, [6])
    value_tensor = helper.make_tensor_value_info('value', TensorProto.FLOAT, [])

    # Pads = [N_before, C_before, W_before, N_after, C_after, W_after]
    pads = [0, 0, 2, 0, 0, 3]

    pads_initializer = helper.make_tensor(name='pads', data_type=TensorProto.INT64, dims=[6], vals=pads)
    value_initializer = helper.make_tensor(name='value', data_type=TensorProto.FLOAT, dims=[], vals=[0.0])

    pad_node = helper.make_node(
        'Pad', name='const_pad', inputs=['global_in', 'pads', 'value'], outputs=['global_out'], mode='constant'
    )

    graph = helper.make_graph(
        nodes=[pad_node],
        name='Pad1DGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[pads_initializer, value_initializer],
        value_info=[pads_tensor, value_tensor],
    )

    model = helper.make_model(graph)

    return model


def test_constantpad_1d():
    class Pad1DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((2, 3), 0)  # pad 2 left, 3 right

        def forward(self, x):
            return self.pad(x)

    model = Pad1DModel()
    model.eval()
    config_pytorch = config_from_pytorch_model(model, (2, 4), channels_last_conversion='off')
    hls_model_pytorch = convert_from_pytorch_model(
        model, output_dir=str(test_root_path / 'hls4mlprj_constpad_1d/pytorch'), hls_config=config_pytorch
    )

    hls_model_pytorch.compile()

    pad1d_onnx = _make_constantpad_onnx_1d()

    config_onnx = config_from_onnx_model(pad1d_onnx)
    hls_model_onnx = convert_from_onnx_model(
        pad1d_onnx, output_dir=str(test_root_path / 'hls4mlprj_constpad_1d/onnx'), hls_config=config_onnx
    )

    hls_model_onnx.compile()

    input_data = np.random.randn(10, 2, 4)
    pred_pytorch = hls_model_pytorch.predict(input_data)
    pred_onnx = hls_model_onnx.predict(input_data)

    np.testing.assert_allclose(pred_pytorch, pred_onnx, rtol=0, atol=1e-5)


def _make_constantpad_onnx_2d():
    input_tensor = helper.make_tensor_value_info('global_in', TensorProto.FLOAT, [1, 2, 3, 4])
    output_tensor = helper.make_tensor_value_info('global_out', TensorProto.FLOAT, [1, 2, 10, 7])
    pads_tensor = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
    value_tensor = helper.make_tensor_value_info('value', TensorProto.FLOAT, [])

    # Pads = [N_before, C_before, H_before, W_before, N_after, C_after, H_after, W_after]
    pads = [0, 0, 3, 1, 0, 0, 4, 2]

    pads_initializer = helper.make_tensor(name='pads', data_type=TensorProto.INT64, dims=[8], vals=pads)
    value_initializer = helper.make_tensor(name='value', data_type=TensorProto.FLOAT, dims=[], vals=[0.0])

    pad_node = helper.make_node(
        'Pad', name='const_pad', inputs=['global_in', 'pads', 'value'], outputs=['global_out'], mode='constant'
    )

    graph = helper.make_graph(
        nodes=[pad_node],
        name='Pad2DGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[pads_initializer, value_initializer],
        value_info=[pads_tensor, value_tensor],
    )

    model = helper.make_model(graph)

    return model


def test_constantpad_2d():
    class Pad2DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad2d((1, 2, 3, 4), 0)  # left, right, top, bottom

        def forward(self, x):
            return self.pad(x)

    model = Pad2DModel()
    model.eval()
    config_pytorch = config_from_pytorch_model(model, (2, 3, 4), channels_last_conversion='off')
    hls_model_pytorch = convert_from_pytorch_model(
        model, output_dir=str(test_root_path / 'hls4mlprj_constpad_2d/pytorch'), hls_config=config_pytorch
    )

    hls_model_pytorch.compile()

    pad2d_onnx = _make_constantpad_onnx_2d()

    config_onnx = config_from_onnx_model(pad2d_onnx)
    hls_model_onnx = convert_from_onnx_model(
        pad2d_onnx, output_dir=str(test_root_path / 'hls4mlprj_constpad_2d/onnx'), hls_config=config_onnx
    )

    hls_model_onnx.compile()

    input_data = np.random.randn(10, 2, 3, 4)
    pred_pytorch = hls_model_pytorch.predict(input_data)
    pred_onnx = hls_model_onnx.predict(input_data)

    np.testing.assert_allclose(pred_pytorch, pred_onnx, rtol=0, atol=1e-5)
