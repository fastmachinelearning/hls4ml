import numpy as np

from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler


@onnx_handler('Conv')
def parse_conv_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    if node.domain != 'qonnx.custom_op.channels_last':
        raise RuntimeError("Please convert the model to channels-last format with qonnx-to-channels-last")
    layer['data_format'] = 'channels_last'  # QONNX needs to be channels-last.
    layer['inputs'] = input_names
    layer['outputs'] = node.output

    strides = get_onnx_attribute(node, 'strides')
    kernel_shape = get_onnx_attribute(node, 'kernel_shape')
    # Note:  currently don't have support for auto_pad.
    pads = get_onnx_attribute(node, 'pads')
    dilations = get_onnx_attribute(node, 'dilations')
    if dilations is None:
        dilations = [1] * len(layer['kernel_shape'])

    layer['in_width'] = input_shapes[0][-2]
    layer['n_chan'] = input_shapes[0][-1]
    layer['n_filt'] = input_shapes[1][0]

    layer['group'] = int(get_onnx_attribute(node, 'group'))
    if layer['group'] != 1:
        layer['depth_multiplier'] = get_onnx_attribute(node, 'group') / layer['n_chan']
        if not layer['depth_multiplier'].is_integer():
            raise ValueError('Depth multiplier must be an integer')
        else:
            layer['depth_multiplier'] = int(layer['depth_multiplier'])

    layer['n_dim'] = len(input_shapes[0]) - 2  # 2 comes from channels and batch dimentions
    if layer['n_dim'] not in (1, 2):
        raise ValueError("Only 1D and 2D convolutions are supported")
    layer['class_name'] = 'Conv'

    # set some values needed later
    if layer['n_dim'] == 1:
        # this is 1D convolution
        full_width = layer['in_width'] + pads[0] + pads[1]
        eff_kernel_width = kernel_shape[0] * dilations[0]
        layer['out_width'] = int(np.ceil((full_width - eff_kernel_width + 1) / strides[0]))
        # for compatibility interpret some variables
        layer['pad_left'] = pads[0]
        layer['pad_right'] = pads[1]
        layer['filt_width'] = kernel_shape[0]
        layer['stride_width'] = strides[0]
        layer['dilation_width'] = dilations[0]
    else:
        # 2d
        layer['in_height'] = input_shapes[0][-3]
        full_height = layer['in_height'] + pads[0] + pads[2]
        eff_kernel_height = kernel_shape[0] * dilations[0]
        out_height = int(np.ceil((full_height - eff_kernel_height + 1) / strides[0]))
        layer['out_height'] = out_height

        full_width = input_shapes[0][-2] + pads[1] + pads[3]
        eff_kernel_width = kernel_shape[1] * dilations[1]
        out_width = int(np.ceil((full_width - eff_kernel_width + 1) / strides[1]))
        layer['out_width'] = out_width
        # for compatibility interpret some variables
        layer['pad_top'] = pads[0]
        layer['pad_left'] = pads[1]
        layer['pad_bottom'] = pads[2]
        layer['pad_right'] = pads[3]
        layer['filt_height'] = kernel_shape[0]
        layer['filt_width'] = kernel_shape[1]
        layer['stride_height'] = strides[0]
        layer['stride_width'] = strides[1]
        layer['dilation_height'] = dilations[0]
        layer['dilation_width'] = dilations[1]

    return layer
