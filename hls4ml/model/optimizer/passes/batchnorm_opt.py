import numpy as np
from hls4ml.model.optimizer import OptimizerPass

class BatchNormConstantParameters(OptimizerPass):
    """ Remove Constant from the BatchNormalization node parameters (but not input[0]) """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'BatchNormalization'
                    and any(node.inputs[1:]))

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the BatchNormalization node parameters (but not input[0])
        """

        if not (len(node.inputs) == 5 and all(node.inputs)):
            raise ValueError(f"All {len.node.inputs} BatchNormalization inputs need to be defined")
        
        gamma_node = node.get_input_node(node.inputs[1])
        if gamma_node.__class__.__name__ != 'Constant':
            raise TypeError("Only consant gammas supported")
        gamma = gamma_node.value
        node.set_attr('gamma', gamma)
        node.inputs[1] = ''
        model.remove_node(gamma_node, rewire=False)

        beta_node = node.get_input_node(node.inputs[2])
        if beta_node.__class__.__name__ != 'Constant':
            raise TypeError("Only consant betas supported")
        beta = beta_node.value
        node.set_attr('beta', beta)
        node.inputs[2] = ''
        model.remove_node(beta_node, rewire=False)

        moving_mean_node = node.get_input_node(node.inputs[3])
        if moving_mean_node.__class__.__name__ != 'Constant':
            raise TypeError("Only consant moving_means supported")
        moving_mean = moving_mean_node.value
        node.set_attr('moving_mean', moving_mean)
        node.inputs[3] = ''
        model.remove_node(moving_mean_node, rewire=False)

        moving_variance_node = node.get_input_node(node.inputs[4])
        if moving_variance_node.__class__.__name__ != 'Constant':
            raise TypeError("Only consant moving_variances supported")
        moving_variance = moving_variance_node.value
        node.set_attr('moving_variance', moving_variance)
        node.inputs[4] = ''
        model.remove_node(moving_variance_node, rewire=False)

        scale = gamma / np.sqrt(moving_variance + node.get_attr('epsilon'))
        bias = beta - gamma * moving_mean / np.sqrt(moving_variance + node.get_attr('epsilon'))

        node.add_weights_variable("scale", data=scale, precision=node.get_attr("quant_precision"), quantizer=node.get_attr("quantizer"))
        node.add_weights_variable("bias", data=bias, precision=node.get_attr("quant_precision"), quantizer=node.get_attr("quantizer"))

        return True


class ConstantBatchNormMerging(OptimizerPass):
    """
    Merge BatchNorm into Const (after parameters have already been merged in BatchNormalization)
    """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'BatchNormalization'
                    and not any(node.inputs[1:])
                    and node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant')

        return is_match
    
    def transform(self, model, node):
        """
        Remove the batch norm
        """
        const_node = node.get_input_node(node.inputs[0])

        new_val = const_node.value * node.weights["scale"].data_unquantized + node.weights["bias"].data_unquantized
        quantizer = node.get_attr("quantizer")  # None if not defined
        if quantizer:
            const_node.set_attr("quantizer", quantizer)
        const_node.set_attr("value", new_val)

        quant_precision = node.get_attr("quant_precision")
        if quant_precision:
            const_node.set_attr("quant_precision", quant_precision)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node.initialize()

        # remove the batch norm node
        model.remove_node(node, rewire=True)
       
        return True


class FuseConsecutiveBatchNormalization(OptimizerPass):
    '''
    OptimizerPass to merge consecutive BatchNormalization layers,
    only if the earlier one does not have quantization specified
    '''

    def match(self, node):
        return (node.__class__.__name__ == 'BatchNormalization'
                and node.get_input_node(node.inputs[0]).__class__.__name__ == 'BatcnNormalization'
                and not node.get_input_node(node.inputs[0]).get_attr("quant_precision"))
 

    def transform(self, model, node):
        prev_node = node.get_input_node()

        s0 = prev_node.weights['scale'].data_unquantized
        b0 = prev_node.weights['bias'].data_unquantized
        s1 = node.weights['scale'].data_unquantized
        b1 = node.weights['bias'].data_unquantized

        scale_new = s0 * s1
        bias_new = s1 * b0 + b1

        # call function so that quantizer would be called if needed
        node.add_weights_variable(name='scale', data=scale_new, precision=node.get_attr("quant_precision"), quantizer=node.get_attr("quantizer"))
        node.add_weights_variable(name='bias', data=bias_new, precision=node.get_attr("quant_precision"), quantizer=node.get_attr("quantizer"))
 
        model.remove_node(prev_node, rewire=True)
        return True
