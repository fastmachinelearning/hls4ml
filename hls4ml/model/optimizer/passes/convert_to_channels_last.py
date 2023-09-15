# Conversion of model from channels_first to channels_last data format
# Based on https://github.com/fastmachinelearning/qonnx/blob/
# 12c96a3ded06beacab08e0f554e4ed014476c0aa/src/qonnx/transformation/channels_last.py

from hls4ml.model.layers import Concatenate, Input, Reshape
from hls4ml.model.optimizer import OptimizerPass


class ChannelsLastConverter(OptimizerPass):
    '''Converts a model from channels_first to channels_last data format by transposing the weights of relevant layers
    and adding a transpose layer for the inputs and outputs, if necessary'''

    def match(self, node):
        if not hasattr(node, 'channels_last_converted'):
            return True

    def transform(self, model, node):
        # If this parameter has not been set, this model does not need to be converted
        if 'InputsChannelLast' not in model.config.config['HLSConfig']['Model']:
            node.channels_last_converted = True
            return False
        outshape = node.get_output_variable().shape

        if isinstance(node, Input):
            # if inputs are not yet transposed into channels_last, add transpose layer
            if not model.config.config['HLSConfig']['Model']['InputsChannelLast'] and len(outshape) > 1:
                # Add transpose for input layer
                input = node.name
                if len(outshape) == 2:
                    attributes = {'perm': [1, 0]}
                else:
                    attributes = {'perm': [1, 2, 0]}

                transpose_node = model.make_node(
                    'Transpose', f'transpose_input_for_{node.get_attr("name")}', attributes, [input]
                )
                transpose_node.set_attr('name', f'transpose_input_for_{node.get_attr("name")}')
                transpose_node.channels_last_converted = True

                model.insert_node(transpose_node)
            else:
                input_shape = node.get_output_variable().shape
                input_shape.append(input_shape.pop(0))
                node.get_output_variable().shape = input_shape
                dim_names = [f'N_INPUT_{i}_{node.index}' for i in range(1, len(input_shape) + 1)]
                node.get_output_variable().dim_names = dim_names
        else:
            # Transpose weight tensors
            tensors = ['weight', 'depthwise', 'pointwise', 'zero_bias', 'scale', 'recurrent_weight']
            for tensor in tensors:
                try:
                    if len(node.get_weights(tensor).shape) == 2:
                        weights_channels_last = node.get_weights(tensor).data.transpose()
                        node.get_weights(tensor).data = weights_channels_last
                    elif len(node.get_weights(tensor).shape) == 3:
                        weights_channels_last = node.get_weights(tensor).data.transpose([2, 1, 0])
                        node.get_weights(tensor).data = weights_channels_last
                    elif len(node.get_weights(tensor).shape) == 4:
                        weights_channels_last = node.get_weights(tensor).data.transpose([2, 3, 1, 0])
                        node.get_weights(tensor).data = weights_channels_last
                except KeyError:
                    pass
            try:
                node.set_attr('data_format', 'channels_last')
            except AttributeError:
                pass

            # Adjust axis of operation
            if isinstance(node, Concatenate):
                old_axis = node.get_attr('axis')
                if len(outshape) == 2:
                    if old_axis == -1 or old_axis == 2:
                        node.set_attr('axis', 1)
                    else:
                        node.set_attr('axis', 2)
                elif len(outshape) == 3:
                    if old_axis == 3 or old_axis == -1:
                        node.set_attr('axis', 1)
                    elif old_axis == 2 or old_axis == -2:
                        node.set_attr('axis', 2)  # Not required, but left for clarity
                    else:
                        node.set_attr('axis', 3)

            # Adjust output shape
            outdims = node.get_output_variable().dim_names
            if len(outshape) == 2:
                shape = [outshape[1], outshape[0]]
                dims = [outdims[1], outdims[0]]
                node.add_output_variable(shape, dims)
            elif len(outshape) == 3:
                shape = [outshape[1], outshape[2], outshape[0]]
                dims = [outdims[1], outdims[2], outdims[0]]
                node.add_output_variable(shape, dims)

            # Have to transpose back before flattening to get correct order of elements in the flattened tensor
            if isinstance(node, Reshape) and len(node.attributes['target_shape']) == 1:
                previous_node = node.get_input_node(node.inputs[0])
                input = previous_node.name
                outshape = previous_node.get_output_variable().shape

                if len(outshape) == 2:
                    attributes = {'perm': [1, 0]}
                else:
                    attributes = {'perm': [2, 0, 1]}

                transpose_node = model.make_node(
                    'Transpose', f'transpose_input_for_{node.get_attr("name")}', attributes, [input]
                )
                transpose_node.channels_last_converted = True

                model.insert_node(transpose_node)

            # Add transpose for output layer
            elif (
                node.get_attr('name') in model.outputs
                and len(outshape) > 1
                and model.config.config['HLSConfig']['Model']['TransposeOutputs']
            ):
                input = node.name
                outshape = node.get_output_variable().shape

                if len(outshape) == 2:
                    attributes = {'perm': [1, 0]}
                else:
                    attributes = {'perm': [2, 0, 1]}

                transpose_node = model.make_node(
                    'Transpose', f'transpose_ouput_for_{node.get_attr("name")}', attributes, [input]
                )
                transpose_node.channels_last_converted = True

                model.insert_node(transpose_node)

        node.channels_last_converted = True
        return True
