from hls4ml.model.optimizer import OptimizerPass

class MergeRelu(OptimizerPass):
    def match(self, node):
        supported_layers = ['Conv2D', 'Conv2DBatchnorm', 'Dense']
        is_match = node.get_input_node().__class__.__name__ in supported_layers

        # hls4ml names ReLU activations 'Activation'
        is_match = is_match and (node.__class__.__name__ == 'Activation') 
        return is_match

    def transform(self, model, node):
        # Merge ReLU and Convolution/Dense layer
        previous_node = node.get_input_node()
        previous_node.index = node.index
        previous_node.set_merged_relu(True) # Turn on merged_relu flag for this Conv/Dense layer
        if 'Conv2D' in previous_node.__class__.__name__:
            if previous_node.get_attr('data_format') == 'channels_last':
                shape = [previous_node.attributes['out_height'], previous_node.attributes['out_width'], previous_node.attributes['n_filt']]
                dims = ['OUT_HEIGHT_{}'.format(previous_node.index), 'OUT_WIDTH_{}'.format(previous_node.index), 'N_FILT_{}'.format(previous_node.index)]
            else:
                shape = [previous_node.attributes['n_filt'], previous_node.attributes['out_height'], previous_node.attributes['out_width']]
                dims = ['N_FILT_{}'.format(previous_node.index), 'OUT_HEIGHT_{}'.format(previous_node.index), 'OUT_WIDTH_{}'.format(previous_node.index)]
            activation_precision, _ = model.config.get_precision(node, var='result')
            previous_node.add_output_variable(shape, dims, precision=activation_precision)
            if not node.get_output_nodes():
                print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
                model.remove_node(node, rewire=False)
            else:
                model.remove_node(node, rewire=True)
            return True 
        elif 'Dense' in previous_node.__class__.__name__:
            shape = previous_node.get_input_variable().shape[:]
            shape[-1] = previous_node.attributes['n_out']
            if len(shape) > 1:
                dims = ['N_LAYER_{}_{}'.format(i, previous_node.index) for i in range(1, len(shape) + 1)]
            else:
                dims = ['N_LAYER_{}'.format(previous_node.index)]
            print('shape: {}'.format(shape))
            print('dims: {}'.format(dims))
            activation_precision, _ = model.config.get_precision(node, var='result')
            previous_node.add_output_variable(shape, dims, precision=activation_precision)
            if not node.get_output_nodes():
                print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
                model.remove_node(node, rewire=False)
            else:
                model.remove_node(node, rewire=True)
            return True