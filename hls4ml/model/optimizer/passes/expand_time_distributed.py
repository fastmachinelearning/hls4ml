from hls4ml.model.layers import TimeDistributed
from hls4ml.model.optimizer import OptimizerPass


class ExpandTimeDistributed(OptimizerPass):
    '''Expands TimeDistributed's wrapped layer into the graph and inserts a marker at the end.

    For example, the layer defined as:
        TimeDistributed(Dense(...))
    will be expanded to:
        TimeDistributed(...)
        Dense(...)
        TimeDistributed(...)
    the latter TimeDistributed serving as a marker of the end of the block and it will have "_end" appended to its name.

    Handling flattened hierarchy has advantages of exposing the wrapped layer(s) to the optimizers. Backends may choose
    to undo this after all optimizers have been applied on the wrapped layers.
    '''

    def match(self, node):
        return isinstance(node, TimeDistributed) and not isinstance(node.get_attr('wrapped_layer'), TimeDistributed)

    def transform(self, model, node):
        output_var = node.get_output_variable()

        # Save the real output shape that the end marker will use
        real_output_shape = output_var.shape.copy()

        # Replace the current node's output shape to one time step (the input to the wrapped layer)
        new_output_shape = node.get_input_variable().shape[1:]
        new_output_dims = [dim.replace('OUT_', 'IN_') for dim in output_var.dim_names[1:]]
        output_var.shape = new_output_shape
        output_var.dim_names = new_output_dims

        # Insert the node into the graph after existing TimeDistributed layer
        # (which should pick up the input shape as one time step)
        wrapped_node = self._make_wrapped_node(model, node)
        node.set_attr('wrapped_layer', wrapped_node)
        model.insert_node(wrapped_node)

        # Create a new TimeDistributed layer to serve as a marker. It will have "_end" appended to the name and will
        # point to itself in wrapped_layer attribute
        new_td_name = node.get_attr('name') + '_end'
        new_td_attr = {
            'wrapped_layer': node,
            'output_shape': real_output_shape,
            'n_time_steps': node.get_attr('n_time_steps'),
        }
        new_td_node = model.make_node(TimeDistributed, new_td_name, new_td_attr, wrapped_node.outputs.copy(), [new_td_name])
        model.insert_node(new_td_node)

        # Set the attribute of the first TimeDistributed to the end marker, so we know the transformation is done
        node.set_attr('wrapped_layer', new_td_node)

        return True

    def _make_wrapped_node(self, model, parent_node):
        # At this stage, the wrapped_layer attribute is a dict (an unprocessed layer) coming from the frontend converter
        layer_proto = parent_node.attributes['wrapped_layer']
        kind = layer_proto['class_name']
        name = layer_proto['name']
        inputs = layer_proto.get('inputs', [])
        outputs = layer_proto.get('outputs', [])
        if len(inputs) == 0:
            inputs = [parent_node.attributes['name']]
        if len(outputs) == 0:
            outputs = [name]

        wrapped_node = model.make_node(kind, name, layer_proto, inputs, outputs)

        return wrapped_node
