import numpy as np

from hls4ml.model.layers import Constant, Conv, Conv1D, Conv2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import IntegerPrecisionType

_base_attributes = (
    'Trace',
    'reuse_factor',
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
    'strategy',
    'data_format',
)


class ConvToConvXD(OptimizerPass):
    """Convert Conv with constant to a Conv1D or Conv2D layer"""

    def match(self, node):
        is_match = isinstance(node, Conv) and (
            (len(node.inputs) == 2 and isinstance(node.get_input_node(node.inputs[1]), Constant))
            or (
                len(node.inputs) == 3
                and isinstance(node.get_input_node(node.inputs[1]), Constant)
                and isinstance(node.get_input_node(node.inputs[2]), Constant)
            )
        )

        return is_match

    def transform(self, model, node):
        """Convert Conv with constant to a Conv1D or Conv2D layer"""

        weight_node = node.get_input_node(node.inputs[1])
        weight_precision = weight_node.get_attr("quant_precision")
        bias_node = None
        bias_precision = None
        if len(node.inputs) == 3:
            bias_node = node.get_input_node(node.inputs[2])
            bias_precision = bias_node.get_attr("quant_precision")

        # creating the attributes
        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}

        # The ConvxD nodes expect the weight data to be in a different format, not (M, k1.., C)
        if node.attributes['n_dim'] == 1:
            newtype = Conv1D
            attributes["weight_data"] = np.transpose(weight_node.value, (1, 2, 0))
        else:
            newtype = Conv2D
            attributes["weight_data"] = np.transpose(weight_node.value, (1, 2, 3, 0))
        attributes["weight_precision"] = weight_precision
        attributes["weight_quantizer"] = weight_node.get_attr("quantizer")

        if bias_node:
            attributes["bias_data"] = (bias_node.value,)
            attributes["bias_precision"] = (bias_precision,)
            attributes["bias_quantizer"] = bias_node.get_attr("quantizer")
        else:
            attributes["bias_data"] = np.zeros(attributes['n_filt'])
            attributes["bias_precision"] = IntegerPrecisionType(1, False)

        # making new node
        new_node = model.make_node(
            newtype, f"{newtype.__name__}_{node.name}", attributes, [node.inputs[0]], [x for x in node.outputs]
        )

        # removing and replacing old nodes
        model.remove_node(weight_node, rewire=False)
        if bias_node:
            model.remove_node(bias_node, rewire=False)
        model.replace_node(node, new_node)

        return True
