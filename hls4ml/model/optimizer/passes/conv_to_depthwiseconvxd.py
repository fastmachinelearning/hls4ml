import numpy as np

from hls4ml.model.layers import Constant, Conv, DepthwiseConv1D, DepthwiseConv2D
from hls4ml.model.optimizer import OptimizerPass

# these are attributes to copy
_base_attributes = (
    'in_width',
    'out_width',
    'n_chan',
    'n_filt',
    'pad_left',
    'pad_right',
    'filt_width',
    'stride_width',
    'dilation_width',
    'in_height',
    'out_height',
    'pad_top',
    'pad_bottom',
    'filt_height',
    'stride_height',
    'dilation_height',
    'data_format',
)


class ConvToDepthwiseConvXD(OptimizerPass):
    """Convert Conv with constant to a DepthwiseConv1D or DepthwiseConv2D layer"""

    def match(self, node):
        is_match = (
            isinstance(node, Conv)
            and node.get_attr('group') == node.get_attr('n_chan')
            and (node.get_attr('group') != 1)
            and (
                (len(node.inputs) == 2 and isinstance(node.get_input_node(node.inputs[1]), Constant))
                or (
                    len(node.inputs) == 3
                    and isinstance(node.get_input_node(node.inputs[1]), Constant)
                    and isinstance(node.get_input_node(node.inputs[2]), Constant)
                )
            )
        )

        return is_match

    def transform(self, model, node):
        """Convert Conv with constant to a DepthwiseConv1D or DepthwiseConv2D layer"""

        weight_node = node.get_input_node(node.inputs[1])
        weight_data = weight_node.attributes['value']
        bias_node = None
        if len(node.inputs) == 3:
            bias_node = node.get_input_node(node.inputs[2])

        # creating the attributes
        attributes = {k: node.attributes[k] for k in _base_attributes if k in node.attributes}

        # The ConvxD nodes expect the weight data to be in a different format, not (M, k1.., C)
        if node.attributes['n_dim'] == 1:
            newtype = DepthwiseConv1D
            attributes['depthwise_data'] = np.transpose(weight_data, (1, 2, 0))
        else:
            newtype = DepthwiseConv2D
            attributes['depthwise_data'] = np.transpose(weight_data, (1, 2, 3, 0))
        attributes['depthwise_quantizer'] = weight_node.get_attr('quantizer')

        if bias_node:
            attributes['bias_data'] = bias_node.attributes['value']
            attributes['bias_quantizer'] = bias_node.get_attr('quantizer')
            attributes['use_bias'] = True
        else:
            attributes['bias_data'] = np.zeros(attributes['n_filt'])
            attributes['use_bias'] = False

        # get the configuration name
        config = model.config.get_layer_config(node)
        new_name = f'{newtype.__name__}_{node.name}'
        model.config.set_name_config(new_name, config)
        model.config.parse_name_config(new_name, config)

        # making new node
        new_node = model.make_node(newtype, new_name, attributes, [node.inputs[0]], [x for x in node.outputs])

        # removing and replacing old nodes
        if bias_node:
            model.remove_node(bias_node)
            del node.inputs[2]
        model.remove_node(weight_node)
        del node.inputs[1]
        model.replace_node(node, new_node)

        return True
