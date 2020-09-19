from hls4ml.model.optimizer import OptimizerPass

class InsertZeroPaddingBeforeConv2D(OptimizerPass):
    def match(self, node):
        is_match = node.__class__.__name__ == 'Conv2D' and \
            node.get_attr('padding') == 'same'
        return is_match

    def transform(self, model, node):
        if model.config.backend.name != 'Vivado' or \
            model.config.get_config_value('IOType') != 'io_stream':
            return False
        
        # Get the padding parameters from Conv2D layer
        pad_top = node.get_attr('pad_top')
        pad_bottom = node.get_attr('pad_bottom')
        pad_left = node.get_attr('pad_left')
        pad_right = node.get_attr('pad_right')

        out_height = pad_top + node.get_attr('in_height') + pad_bottom
        out_width = pad_left + node.get_attr('in_width') + pad_right

        attrs = {
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'in_height': node.get_attr('in_height'),
            'in_width': node.get_attr('in_width'),
            'out_height': out_height,
            'out_width': out_width,
            'n_chan': node.get_attr('n_chan'),
            'data_format': node.get_attr('data_format', 'channels_last')
        }

        # Switch Conv2D layer padding to 'valid'
        node.set_attr('padding', 'valid')
        node.set_attr('pad_top', 0)
        node.set_attr('pad_bottom', 0)
        node.set_attr('pad_left', 0)
        node.set_attr('pad_right', 0)
        node.set_attr('in_height', out_height)
        node.set_attr('in_width', out_width)

        # Insert new ZeroPadding2D node above Conv2D
        padding_layer = model.make_node('ZeroPadding2D', 'zp2d_' + node.name, attrs, node.inputs.copy())
        model.insert_node(padding_layer)

        return True
