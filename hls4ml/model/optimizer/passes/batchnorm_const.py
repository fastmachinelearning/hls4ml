import numpy as np
from hls4ml.model.optimizer import OptimizerPass

class BatchNormConstantParameters(OptimizerPass):
    """ Remove Constant from the BaseBatchNormalization node parameters (but not input[0]) """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'BaseBatchNormalization'
                    and any(node.inputs[1:]))

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the BaseBatchNormalization node parameters (but not input[0])
        """

        if not (len(node.inputs) == 5 and all(node.inputs)):
            raise ValueError(f"All {len.node.inputs} BaseBatchNormalization inputs need to be defined")
        
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

        node.set_attr("scale", scale)
        node.set_attr("bias", bias)
        # Note:  These still need to be written out as variables with a type

        return True


class ConstantBatchNormMerging(OptimizerPass):
    """
    Merge BatchNorm into Const
    """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'BaseBatchNormalization'
                    and not any(node.inputs[1:])
                    and node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant')

        return is_match
    
    def transform(self, model, node):
        """
        Remove the batch norm
        """
        const_node = node.get_input_node(node.inputs[0])

        new_val = const_node.value * node.get_attr("scale") + node.get_attr("bias")
        quantizer = node.get_attr("quantizer")  # None if not defined
        if quantizer:
            # need to quantize the data
            new_val = quantizer(new_val)
            const_node.set_attr("quantizer", quantizer)
        const_node.set_attr("value", new_val)

        quant_precision = node.get_attr("quant_precision")
        if quant_precision:
            const_node.set_attr("quant_precision", quant_precision)

        # reinitialize
        const_node.initialize()

        # remove the batch norm node
        model.remove_node(node, rewire=True)
       
        return True