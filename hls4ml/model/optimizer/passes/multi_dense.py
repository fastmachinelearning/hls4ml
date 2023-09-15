import numpy as np

from hls4ml.model.layers import Dense
from hls4ml.model.optimizer import OptimizerPass


class ReplaceMultidimensionalDenseWithConv(OptimizerPass):
    def match(self, node):
        return (
            isinstance(node, Dense)
            and len(node.get_input_variable().shape) - sum(d == 1 for d in node.get_input_variable().shape) > 1
        )
        # The above sum checks for the number of dimensions in the Dense with size 1
        # The subtraction allows the check to only count the number of dimensions with non-1 size
        # For example, this prevents matching for a Dense layer with shape (1,N)

    def transform(self, model, node):
        dim = len(node.get_input_variable().shape) - 1
        input_shape = node.get_input_variable().shape

        pointwise_attrs = {
            'data_format': 'channels_last',
            'padding': 'valid',
            'n_chan': input_shape[-1],
            'n_filt': node.get_attr('n_out'),
            'weight_data': node.get_attr('weight_data'),
            'bias_data': node.get_attr('bias_data'),
        }

        if dim == 1:
            pointwise_attrs.update(
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
            pointwise_attrs.update(
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

        class_name = 'PointwiseConv' + str(dim) + 'D'
        pw_node = model.make_node(class_name, node.name, pointwise_attrs, node.inputs.copy())
        if len(node.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            pw_node.weights['weight'].data = np.expand_dims(node.weights['weight'].data, axis=tuple(range(dim)))
        pw_node.weights['bias'].data = node.weights['bias'].data
        model.replace_node(node, pw_node)

        return True
