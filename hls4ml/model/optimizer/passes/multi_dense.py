import numpy as np

from hls4ml.model.layers import Dense
from hls4ml.model.optimizer import OptimizerPass


class ReplaceMultidimensionalDenseWithConv(OptimizerPass):
    """
    This matches all multidimensional Dense layers and changes them to a convolution.
    Note:  the convolution may subsequently be changed to a pointwise convolution for
    bakends that implement special pointwise convolutions.
    """

    def match(self, node):
        return isinstance(node, Dense) and len(node.get_input_variable().shape) > 1

    def transform(self, model, node):
        dim = len(node.get_input_variable().shape) - 1
        input_shape = node.get_input_variable().shape

        conv_attrs = {
            'data_format': 'channels_last',
            'n_chan': input_shape[-1],
            'n_filt': node.get_attr('n_out'),
            'weight_data': np.expand_dims(node.get_attr('weight_data'), axis=tuple(range(dim))),
            'bias_data': node.get_attr('bias_data'),
        }

        if (pf := node.get_attr('parallelization_factor', None)) is not None:
            conv_attrs['parallelization_factor'] = pf

        if dim == 1:
            conv_attrs.update(
                {
                    'in_width': input_shape[0],
                    'out_width': input_shape[0],
                    'filt_width': 1,
                    'stride_width': 1,
                    'pad_left': 0,
                    'pad_right': 0,
                }
            )
        elif dim == 2:
            conv_attrs.update(
                {
                    'in_height': input_shape[0],
                    'in_width': input_shape[1],
                    'out_height': input_shape[0],
                    'out_width': input_shape[1],
                    'filt_height': 1,
                    'filt_width': 1,
                    'stride_height': 1,
                    'stride_width': 1,
                    'pad_top': 0,
                    'pad_bottom': 0,
                    'pad_left': 0,
                    'pad_right': 0,
                }
            )
        else:
            raise Exception('Cannot replace Dense over {dim}D tensor with Conv{dim}D.'.format(dim=dim))

        class_name = 'Conv' + str(dim) + 'D'
        conv_node = model.make_node(class_name, node.name, conv_attrs, node.inputs.copy())
        model.replace_node(node, conv_node)

        return True
