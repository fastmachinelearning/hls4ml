from hls4ml.converters.pytorch_to_hls import get_weights_data, pytorch_handler
from hls4ml.converters.utils import compute_padding_1d_pytorch, compute_padding_2d_pytorch, parse_data_format
from hls4ml.model.types import FixedPrecisionType
import math

def ConvUAQToAp_Fixed(bitwidth, scale_factor, zero_point):
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
  
  return (fract_bitwidth, int_bitwidth)

@pytorch_handler('Conv1d', 'QuantConv1d')
def parse_conv1d_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Conv1d' in operation

    layer = {}

    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    layer['weight_data'] = get_weights_data(data_reader, layer['name'], 'weight')
    layer['bias_data'] = get_weights_data(data_reader, layer['name'], 'bias')
    # Input info
    (layer['in_width'], layer['n_chan']) = parse_data_format(
        input_shapes[0], 'channels_first'
    )  # Keras's default is channels_last

    # Additional parameters
    layer['n_filt'] = class_object.out_channels
    layer['filt_width'] = class_object.kernel_size[0]
    layer['stride_width'] = class_object.stride[0]
    layer['dilation'] = class_object.dilation[0]

    if type(class_object.padding) is tuple:
        padding = class_object.padding[0]
    else:
        padding = class_object.padding

    if padding == 0:  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else:  # Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'

    # Ouput info
    (layer['out_width'], pad_left, pad_right) = compute_padding_1d_pytorch(
        padding, layer['in_width'], layer['stride_width'], layer['filt_width'], layer['dilation']
    )
    layer['pad_left'] = pad_left
    layer['pad_right'] = pad_right

    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_width']]  # Channel first as default

    return layer, output_shape


@pytorch_handler('Conv2d', 'QuantConv2d')
def parse_conv2d_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Conv2d' in operation

    layer = {}

    layer['name'] = layer_name
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    if "Quant" in operation:
        layer['weight_data'] = class_object.quant_weight().detach().value.numpy()
        layer['bias_data'] = class_object.quant_bias().detach().value.numpy()
        width = class_object.quant_weight().bit_width
        ap_fixed_params = ConvUAQToAp_Fixed(width, class_object.quant_weight().scale,0)
        layer['precision'] = FixedPrecisionType(width=width, integer=ap_fixed_params[1], signed=True)
    
    else:
        layer['weight_data'] = get_weights_data(data_reader, layer['name'], 'weight')
        layer['bias_data'] = get_weights_data(data_reader, layer['name'], 'bias')
    # Input info
    (layer['in_height'], layer['in_width'], layer['n_chan']) = parse_data_format(
        input_shapes[0], 'channels_first'
    )  # Keras's default is channels_last

    # Additional parameters
    layer['n_filt'] = class_object.out_channels
    layer['filt_height'] = class_object.kernel_size[0]
    layer['filt_width'] = class_object.kernel_size[1]
    layer['stride_height'] = class_object.stride[0]
    layer['stride_width'] = class_object.stride[1]
    layer['dilation'] = class_object.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = class_object.padding[0]
    layer['pad_left'] = layer['pad_right'] = class_object.padding[1]

    if all(x == 0 for x in class_object.padding):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else:  # Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'

    # Ouput info
    (layer['out_height'], layer['out_width'], _, _, _, _) = compute_padding_2d_pytorch(
        class_object.padding,
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width'],
        class_object.dilation[0],
        class_object.dilation[1],
    )

    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
