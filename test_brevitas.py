import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

import math




"""
conversion function from target ap_fixed to parameters for UAQ
"""
from typing import Tuple

def ConvAp_FixedToUAQ(int_bitwidth, fract_bitwidth) -> Tuple[int,float,float]:
  """
  parameters:
  int_bitwidth: int 
  fract_bitwidth: int

  return:
  bitwidth: int
  scale_factor: float
  zero_point: float
  """
  bitwidth = int_bitwidth + fract_bitwidth
  scale_factor = 2**(-fract_bitwidth)
  zero_point = 0 # we assume int representation is signed
  
  return (bitwidth, scale_factor, zero_point)
     


"""
conversion function from UAQ to ap_fixed
"""
from typing import Tuple

def ConvUAQToAp_Fixed(bitwidth, scale_factor, zero_point) -> Tuple[int,int]:
  """
  parameters:
  bitwidth: int
  scale_factor: float
  zero_point: float
  
  return:
  int_bitwidth: int 
  fract_bitwidth: int
  """
  fract_bitwidth = - math.log2(scale_factor)
  int_bitwidth = bitwidth - fract_bitwidth 
  
  return (bitwidth, int_bitwidth)
     


class QuantWeightLeNet(Module):
    def __init__(self):
        super(QuantWeightLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4)
        self.relu2 = nn.ReLU()
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = nn.ReLU()
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = nn.ReLU()
        self.fc3   = qnn.QuantLinear(84, 10, bias=True, weight_bit_width=4)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        #out = out.reshape(out.reshape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

quant_weight_lenet = QuantWeightLeNet()


class QuantModel(Module):
    def __init__(self):
        super(QuantModel, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        #self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out

quant_weight_lenet = QuantWeightLeNet()

model = QuantModel()


x = torch.randn(3,6,5)

quant_linear = qnn.QuantLinear(2, 4, weight_quant=Int8WeightPerTensorFixedPoint, bias=False)
print(f"Weight QuantTensor Linear:\n {quant_linear.quant_weight()}")
print(f"Quant Weight fix point: {- math.log2(quant_linear.quant_weight().scale)}")
print(f"Quant Weight scale: {quant_linear.quant_weight().scale}")
print(f"Quant Weight bit width: {quant_linear.quant_weight().bit_width}")
print(f"Quant Weight zero point: {quant_linear.quant_weight().zero_point}")

pytorch_prediction = model(x).detach().numpy()
print(f"Weight Tensor:\n {model.conv1.weight}")
print(f"Weight QuantTensor:\n {model.conv1.quant_weight()}")
print(f"Quant Weight fix point: {- math.log2(model.conv1.quant_weight().scale)}")
print(f"Quant Weight scale: {model.conv1.quant_weight().scale}")
print(f"Quant Weight bit width: {model.conv1.quant_weight().bit_width}")
print(f"Quant Weight zero point: {model.conv1.quant_weight().zero_point}")
ap_fixed_params = ConvUAQToAp_Fixed(8, model.conv1.quant_weight().scale,0)
print (ap_fixed_params)
config = config_from_pytorch_model(model, inputs_channel_last=False,transpose_outputs=True)
#config['Model']['Precision'] = 'ap_fixed<%d,%d>'%(ap_fixed_params[0],ap_fixed_params[1])
print (config)
output_dir = "test_pytorch"
backend = "Vivado"
io_type = 'io_parallel'

hls_model = convert_from_pytorch_model(
    model,
    (None, 3,6,5),
    hls_config=config,
    output_dir=output_dir,
    backend=backend,
    io_type=io_type,
)
hls_model.compile()

hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)
print(pytorch_prediction)
print(hls_prediction)