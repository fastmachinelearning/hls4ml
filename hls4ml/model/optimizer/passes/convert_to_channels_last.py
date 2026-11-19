# Conversion of model from channels_first to channels_last data format
# Based on https://github.com/fastmachinelearning/qonnx/blob/
# 12c96a3ded06beacab08e0f554e4ed014476c0aa/src/qonnx/transformation/channels_last.py

from collections import OrderedDict

import numpy as np

from hls4ml.model.layers import GRU, LSTM, Concatenate, Dense, Input, LayerNormalization, Reshape, Transpose
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer
from hls4ml.model.types import WeightVariable


def insert_transpose_after(model, node, name, perm, rewire_consumers=True, rewire_output=True):
    """Insert a Transpose of the output of ``node`` that feeds all layers consuming that output, and that replaces it
    as a model output. Either can be disabled. Nothing is inserted, and None returned, if there is nothing to rewire."""
    out_name = node.outputs[0]
    consumers = [layer for layer in model.graph.values() if out_name in layer.inputs] if rewire_consumers else []
    replaces_output = rewire_output and out_name in model.outputs
    if not consumers and not replaces_output:
        return None

    transpose_node = model.make_node('Transpose', name, {'perm': perm}, [out_name])
    transpose_node.channels_last_converted = True
    for layer in consumers:
        layer.inputs = [transpose_node.outputs[0] if inp == out_name else inp for inp in layer.inputs]
    if replaces_output:
        model.replace_output(out_name, transpose_node.outputs[0])

    new_graph = OrderedDict()
    for layer_name, layer in model.graph.items():
        new_graph[layer_name] = layer
        if layer_name == node.name:
            new_graph[transpose_node.name] = transpose_node
    model.graph = new_graph
    return transpose_node


class ChannelsLastConverter(OptimizerPass):
    """Converts a model from channels_first to channels_last data format by transposing the weights of relevant layers
    and adding a transpose layer for the inputs and outputs, if necessary"""

    def match(self, node):
        # If this parameter has not been set, this model does not need to be converted
        do_convert = node.model.config.config['HLSConfig']['Model'].get('ChannelsLastConversion', 'off')
        if do_convert == 'off':
            return False
        if not hasattr(node, 'channels_last_converted'):
            return True

    def transform(self, model, node):
        outshape = node.get_output_variable().shape

        if isinstance(node, Input):
            # if inputs are not yet transposed into channels_last, add transpose layer
            if model.config.config['HLSConfig']['Model']['ChannelsLastConversion'] == 'full' and len(outshape) > 1:
                # Add transpose for input layer
                perm = [1, 0] if len(outshape) == 2 else [1, 2, 0]
                transpose_node = insert_transpose_after(model, node, f'transpose_input_for_{node.name}', perm)
                if transpose_node is not None:
                    transpose_node.set_attr('name', transpose_node.name)
            elif model.config.config['HLSConfig']['Model']['ChannelsLastConversion'] == 'internal' and len(outshape) > 1:
                input_shape = node.get_output_variable().shape
                input_shape.append(input_shape.pop(0))
                node.get_output_variable().shape = input_shape
        elif isinstance(node, LayerNormalization) or (isinstance(node, Dense) and len(node.get_input_variable().shape) > 1):
            # LayerNorm and Linear act on the last dimension in PyTorch, so transpose the input back to channels_first
            if isinstance(node, Dense):
                # weight_data is also transposed, as ReplaceMultidimensionalDenseWithConv builds the conv from it
                node.set_attr('weight_data', node.get_attr('weight_data').transpose())
                node.get_weights('weight').data = node.get_weights('weight').data.transpose()
            if len(outshape) == 2:
                pre_perm, post_perm = [1, 0], [1, 0]
            else:
                pre_perm, post_perm = [2, 0, 1], [1, 2, 0]
            pre_transpose = model.make_node(
                'Transpose', f'pre_transpose_for_{node.name}', {'perm': pre_perm}, [node.inputs[0]]
            )
            pre_transpose.channels_last_converted = True
            model.insert_node(pre_transpose, before=node)

            # Restore channels_last, except for a model output that should stay channels_first
            insert_transpose_after(
                model,
                node,
                f'post_transpose_for_{node.name}',
                post_perm,
                rewire_output=not model.config.config['HLSConfig']['Model']['TransposeOutputs'],
            )
        elif isinstance(node, LSTM) or isinstance(node, GRU):
            pass
        else:
            if isinstance(node, FixedPointQuantizer):
                transpose_map = {3: (0, 2, 1), 4: (0, 2, 3, 1), 5: (0, 2, 3, 4, 1)}
                node.mask_kbi = tuple(
                    np.transpose(t, transpose_map[t.ndim]) if t.ndim in transpose_map else t for t in node.mask_kbi
                )
            else:
                # Transpose weight tensors
                tensors = ['weight', 'depthwise', 'pointwise', 'zero_bias', 'scale', 'recurrent_weight']
                for tensor in tensors:
                    try:
                        t_shape = node.get_weights(tensor).shape
                        if len(t_shape) == 2:
                            weights_channels_last = node.get_weights(tensor).data.transpose()
                            node.get_weights(tensor).data = weights_channels_last
                        elif len(t_shape) == 3:
                            weights_channels_last = node.get_weights(tensor).data.transpose([2, 1, 0])
                            node.get_weights(tensor).data = weights_channels_last
                        elif len(t_shape) == 4:
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
            if len(outshape) == 2:
                shape = [outshape[1], outshape[0]]
                node.add_output_variable(shape)
            elif len(outshape) == 3:
                shape = [outshape[1], outshape[2], outshape[0]]
                node.add_output_variable(shape)

            # Have to transpose back before flattening to get correct order of elements in the flattened tensor
            if (
                isinstance(node, Reshape)
                and len(node.attributes['target_shape']) == 1
                and not model.config.config['HLSConfig']['Model']['ChannelsLastConversion'] == 'off'
            ):
                outshape = node.get_input_variable().shape

                if len(outshape) == 2:
                    attributes = {'perm': [1, 0]}
                else:
                    attributes = {'perm': [2, 0, 1]}

                transpose_node = model.make_node(
                    'Transpose', f'transpose_input_for_{node.get_attr("name")}', attributes, [node.inputs[0]]
                )
                transpose_node.channels_last_converted = True

                model.insert_node(transpose_node, before=node)

            # Add transpose for output layer
            elif (
                node.get_attr('name') in model.outputs
                and len(outshape) > 1
                and model.config.config['HLSConfig']['Model']['TransposeOutputs']
            ):
                perm = [1, 0] if len(node.get_output_variable().shape) == 2 else [2, 0, 1]
                insert_transpose_after(
                    model, node, f'transpose_ouput_for_{node.get_attr("name")}', perm, rewire_consumers=False
                )

        node.channels_last_converted = True
        return True


class RemoveTransposeBeforeFlatten(OptimizerPass):
    """After the channels last conversion, model may have a sequence: Transpose -> Flatten -> Dense.
    In this case we can remove the expensive transpose and instead transpose the weights of the Dense layer."""

    def match(self, node):
        if node.model.config.get_config_value('IOType') != 'io_parallel':
            return False

        if hasattr(node, '_channels_last_keep_transpose') and node._channels_last_keep_transpose:
            return False

        if isinstance(node, Reshape):
            input_node = node.get_input_node()
            output_nodes = node.get_output_nodes()
            if (
                len(node.get_attr('target_shape')) == 1
                and isinstance(input_node, Transpose)
                and len(output_nodes) == 1
                and isinstance(output_nodes[0], Dense)
            ):
                return True

        return False

    def transform(self, model, node):
        transpose_node = node.get_input_node()
        dense_node = node.get_output_nodes()[0]
        input_shape = transpose_node.get_output_variable().shape

        if len(input_shape) == 2:  # Usually after Conv1D
            tran_axis = [1, 0, 2]
        elif len(input_shape) == 3:  # Usually after Conv2D
            tran_axis = [1, 2, 0, 3]
        else:  # In this case we bail
            node._channels_last_keep_transpose = True
            return False

        weight_var = dense_node.get_weights('weight')
        # Transpose the weights to achieve the same computation with transposed input
        weight_data_t = weight_var.data.reshape(*input_shape, -1).transpose(*tran_axis)
        weight_data_t = weight_data_t.reshape(-1, weight_data_t.shape[-1])
        new_weight_var = WeightVariable(
            var_name=weight_var.name,
            type_name=weight_var.type.name,
            precision=weight_var.type.precision,
            quantizer=weight_var.quantizer,
            data=weight_data_t,
            index=dense_node.index,
        )

        # Update the weight variable of the node
        dense_node.set_attr('weight', new_weight_var)

        # Get rid of the Transpose node
        model.remove_node(transpose_node)

        return True
