from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Conv1D, SeparableConv1D, Conv2D, SeparableConv2D, Conv1DTranspose, Conv2DTranspose

class InsertZeroPaddingBeforeConv1D(OptimizerPass):
    name = 'insert_zero_padding_before_conv1d'
    
    def match(self, node):
        is_match = isinstance(node, (Conv1D, SeparableConv1D)) and \
            ((node.get_attr('padding') == 'same') or (node.get_attr('padding') == 'causal')) and \
            node.get_attr('filt_width') != 1
        return is_match

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False
        
        # Get the padding parameters from Conv1D layer
        pad_left = node.get_attr('pad_left')
        pad_right = node.get_attr('pad_right')

        # Check if no padding needs to be done
        if pad_left == pad_right == 0:
            return False

        out_width = pad_left + node.get_attr('in_width') + pad_right

        attrs = {
            'pad_left': pad_left,
            'pad_right': pad_right,
            'in_width': node.get_attr('in_width'),
            'out_width': out_width,
            'n_chan': node.get_attr('n_chan'),
            'data_format': node.get_attr('data_format', 'channels_last')
        }

        # Switch Conv1D layer padding to 'valid'
        node.set_attr('padding', 'valid')
        node.set_attr('pad_left', 0)
        node.set_attr('pad_right', 0)
        node.set_attr('in_width', out_width)

        # Insert new ZeroPadding1D node above Conv1D
        padding_layer = model.make_node('ZeroPadding1D', 'zp1d_' + node.name, attrs, node.inputs.copy())
        padding_layer.get_output_variable().type.precision = node.get_input_variable().type.precision
        model.insert_node(padding_layer)

        return True

class InsertZeroPaddingBeforeConv1DTranspose(OptimizerPass):
    name = 'insert_zero_padding_before_conv1dtranspose'
    
    def match(self, node):
        is_match = isinstance(node, (Conv1DTranspose)) and \
            node.get_attr('padding') == 'same' and \
            node.get_attr('filt_width') != 1
        return is_match

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False
        
        # Get the padding parameters from Conv1D layer
        pad_left = node.get_attr('pad_left')
        pad_right = node.get_attr('pad_right')
        convtr_out_width = node.get_attr('out_width')
        in_width = node.get_attr('in_width')
        stride_width = node.get_attr('stride_width')
        trfilt_width = (node.get_attr('filt_width') + node.get_attr('stride_width') - 1) \
            // node.get_attr('stride_width')

        add_right = (convtr_out_width + pad_left)//stride_width - (in_width-1)

        out_width = in_width + add_right + trfilt_width-1

        attrs = {
            'pad_left': trfilt_width-1,
            'pad_right': add_right,
            'in_width': in_width,
            'out_width': out_width,
            'n_chan': node.get_attr('n_chan'),
            'data_format': node.get_attr('data_format', 'channels_last')
        }

        # Switch Conv1DTranspose to be 'valid'. I think this is wrong
        node.set_attr('padding', 'valid')
        node.set_attr('in_width', out_width)
        node.set_attr('pad_left', pad_left + (trfilt_width-1)*stride_width)

        # Insert new ZeroPadding1D node above Conv1DTranspose
        padding_layer = model.make_node('ZeroPadding1D', 'zp1d_' + node.name, attrs, node.inputs.copy())
        padding_layer.get_output_variable().type.precision = node.get_input_variable().type.precision
        model.insert_node(padding_layer)

        return True

class InsertZeroPaddingBeforeConv2D(OptimizerPass):
    name = 'insert_zero_padding_before_conv2d'

    def match(self, node):
        is_match = isinstance(node, (Conv2D, SeparableConv2D)) and \
            node.get_attr('padding') == 'same' and \
            node.get_attr('filt_height') != 1 and node.get_attr('filt_width') != 1
        return is_match

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False
        
        # Get the padding parameters from Conv2D layer
        pad_top = node.get_attr('pad_top')
        pad_bottom = node.get_attr('pad_bottom')
        pad_left = node.get_attr('pad_left')
        pad_right = node.get_attr('pad_right')

        # Check if no padding neeeds to be done
        if pad_top == pad_bottom == pad_left == pad_right == 0:
            return False

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
        padding_layer.get_output_variable().type.precision = node.get_input_variable().type.precision
        model.insert_node(padding_layer, before=node)

        return True

class InsertZeroPaddingBeforeConv2DTranspose(OptimizerPass):
    name = 'insert_zero_padding_before_conv2dtranspose'

    def match(self, node):
        is_match = isinstance(node, Conv2DTranspose) and \
            node.get_attr('padding') == 'same' and \
            node.get_attr('filt_width') != 1
        return is_match
    
    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False
        
        # Get the padding parameters from Conv2DTranspose layer
        pad_left = node.get_attr('pad_left')
        pad_right = node.get_attr('pad_right')
        pad_top = node.get_attr('pad_top')
        pad_bottom = node.get_attr('pad_bottom')
        convtr_out_width = node.get_attr('out_width')
        convtr_out_height = node.get_attr('out_height')
        in_width = node.get_attr('in_width')
        in_height = node.get_attr('in_height')
        stride_width = node.get_attr('stride_width')
        stride_height = node.get_attr('stride_height')
        trfilt_width = (node.get_attr('filt_width') + node.get_attr('stride_width') - 1) \
             // node.get_attr('stride_width')
        trfilt_height = (node.get_attr('filt_height') + node.get_attr('stride_height') - 1) \
             // node.get_attr('stride_height')

        add_right = (convtr_out_width + pad_left)//stride_width-(in_width-1)
        add_bottom = (convtr_out_height + pad_top)//stride_height-(in_height-1)

        out_width = in_width + add_right + trfilt_width-1
        out_height = in_height + add_bottom + trfilt_height-1

        attrs = {
            'pad_left': trfilt_width-1,
            'pad_right': add_right,
            'pad_top': trfilt_height-1,
            'pad_bottom': add_bottom,
            'in_width': in_width,
            'in_height': in_height,
            'out_width': out_width,
            'out_height': out_height,
            'n_chan': node.get_attr('n_chan'),
            'data_format': node.get_attr('data_format', 'channels_last')
        }

        # switch Conv2DTranspose to be 'valid'. This is technically not true though
        node.set_attr('padding', 'valid')
        node.set_attr('in_width', out_width)
        node.set_attr('in_height', out_height)
        node.set_attr('pad_left', pad_left + (trfilt_width-1)*stride_width)
        node.set_attr('pad_top', pad_top + (trfilt_height-1)*stride_height)

        # insert new ZeroPadding2D ndoe above Conv2DTranspose
        padding_layer = model.make_node('ZeroPadding2D', 'zp2d_' + node.name, attrs, node.inputs.copy())
        padding_layer.get_output_variable().type.precision = node.get_input_variable().type.precision
        model.insert_node(padding_layer, before=node)

        return True

