from hls4ml.model.optimizer import OptimizerPass

class MergeRelu(OptimizerPass):
    def match(self, node):
        supported_layers = ['Conv2D', 'Conv2DBatchnorm', 'Dense']
        is_match = node.get_input_node().__class__.__name__ in supported_layers

        #hls4ml names ReLU activations 'Activation'
        is_match = is_match and (node.__class__.__name__ == 'Activation') 
        return is_match

    def transform(self, model, node):
        #Merge ReLU and Convolution layer if needed
        previous_node = node.get_input_node()
        previous_node.index = node.index
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