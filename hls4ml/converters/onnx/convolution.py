from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute, compute_pads_1d, compute_pads_2d
from hls4ml.converters.utils import compute_padding_1d, compute_padding_2d

@onnx_handler('Conv')
def parse_conv_layer(reader, node, inputs_map, input_shapes, graph, config):

    layer = {}
    layer['name'] = node.name
    if node.domain != 'qonnx.custom_op.channels_last':
        raise RuntimeError("Please convert the model to channels-last format with qonnx-to-channels-last")
    layer['data_format'] = 'channels_last' # QONNX needs to be channels-last.
    layer['inputs'] = node.input
    layer['outputs'] = node.output
    #reader.add_input(layer['name'], node.input)

    layer['strides'] = get_onnx_attribute(node, 'strides')
    layer['kernel_shape'] = get_onnx_attribute(node, 'kernel_shape')
    # Note:  currently don't have support for auto_pad.
    layer['pads'] = get_onnx_attribute(node, 'pads')
    dilations = get_onnx_attribute(node, 'dilations')
    if dilations is None:
        dilations = [1]*len(layer['kernel_shape'])
    layer['dilations'] = dilations

    if get_onnx_attribute(node, 'group') != 1:
        raise ValueError("Only 1 group supported corrently")

    layer['n_chan'] = input_shapes[0][-1]
    layer['n_dim'] = len(input_shapes[0]) - 2  # 2 comes from channels and batch dimentions
    if layer['n_dim'] not in (1, 2):
        raise ValueError("Only 1D and 2D convolutions are supported")
    layer['class_name'] = 'Conv'

    return layer
