import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model


def test_pytorch_constantpad_1d_2d():
    class Pad1DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((2, 3), 0)  # pad 2 left, 3 right

        def forward(self, x):
            return self.pad(x)

    class Pad2DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad2d((1, 2, 3, 4), 0)  # left, right, top, bottom

        def forward(self, x):
            return self.pad(x)

    # 1D test: batch=1, channels=2, width=4, values 1,2,3,4
    model_1d = Pad1DModel()
    model_1d.eval()
    config_1d = config_from_pytorch_model(model_1d, (2, 4))
    hls_model_1d = convert_from_pytorch_model(model_1d, hls_config=config_1d)
    print("1D Padding Model Layers:")
    for layer in hls_model_1d.get_layers():
        print(f"{layer.name}: {layer.class_name}")

    # 2D test: batch=1, channels=1, height=2, width=4, values 1,2,3,4,5,6,7,8
    model_2d = Pad2DModel()
    model_2d.eval()
    config_2d = config_from_pytorch_model(model_2d, (1, 2, 4))
    hls_model_2d = convert_from_pytorch_model(model_2d, hls_config=config_2d)
    print("2D Padding Model Layers:")
    for layer in hls_model_2d.get_layers():
        print(f"{layer.name}: {layer.class_name}")

    # Write the HLS projects, cannot compile on Windows
    hls_model_1d.write()
    hls_model_2d.write()
