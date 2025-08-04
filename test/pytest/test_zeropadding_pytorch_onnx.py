from pathlib import Path

import numpy as np
import qonnx.util.cleanup
import torch
import torch.nn as nn
from qonnx.core.modelwrapper import ModelWrapper

from hls4ml.converters import convert_from_onnx_model, convert_from_pytorch_model
from hls4ml.utils.config import config_from_onnx_model, config_from_pytorch_model

test_root_path = Path(__file__).parent


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

    onnx_path = str(test_root_path / 'hls4mlprj_constpad_1d/pad1d.onnx')
    torch.onnx.export(model, torch.randn(1, 2, 4), onnx_path, opset_version=10)
    qonnx.util.cleanup.cleanup(onnx_path, out_file=onnx_path)
    pad1d_onnx = ModelWrapper(onnx_path)

    config_onnx = config_from_onnx_model(pad1d_onnx)
    hls_model_onnx = convert_from_onnx_model(
        pad1d_onnx, output_dir=str(test_root_path / 'hls4mlprj_constpad_1d/onnx'), hls_config=config_onnx
    )

    hls_model_onnx.compile()

    input_data = np.random.randn(10, 2, 4)
    pred_pytorch = hls_model_pytorch.predict(input_data)
    pred_onnx = hls_model_onnx.predict(input_data)

    np.testing.assert_allclose(pred_pytorch, pred_onnx, rtol=0, atol=1e-5)


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

    onnx_path = str(test_root_path / 'hls4mlprj_constpad_2d/pad2d.onnx')
    torch.onnx.export(model, torch.randn(1, 2, 3, 4), onnx_path, opset_version=10)
    qonnx.util.cleanup.cleanup(onnx_path, out_file=onnx_path)
    pad2d_onnx = ModelWrapper(onnx_path)

    config_onnx = config_from_onnx_model(pad2d_onnx)
    hls_model_onnx = convert_from_onnx_model(
        pad2d_onnx, output_dir=str(test_root_path / 'hls4mlprj_constpad_2d/onnx'), hls_config=config_onnx
    )

    hls_model_onnx.compile()

    input_data = np.random.randn(10, 2, 3, 4)
    pred_pytorch = hls_model_pytorch.predict(input_data)
    pred_onnx = hls_model_onnx.predict(input_data)

    np.testing.assert_allclose(pred_pytorch, pred_onnx, rtol=0, atol=1e-5)
