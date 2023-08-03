import numpy as np

from hls4ml.model.layers import BatchNormalization, BatchNormOnnx, Constant
from hls4ml.model.optimizer import OptimizerPass

_base_attributes = ('Trace', 'reuse_factor', 'epsilon', 'n_in', 'n_filt')


class BatchNormOnnxConstantParameters(OptimizerPass):
    """Remove Constant from the BatchNormalization node parameters (but not input[0])"""

    def match(self, node):
        is_match = isinstance(node, BatchNormOnnx) and any(node.inputs[1:])

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the BatchNormalization node parameters (but not input[0])
        """

        if not (len(node.inputs) == 5 and all(node.inputs)):
            raise ValueError(f"All {len.node.inputs} BatchNormOnnnx inputs need to be defined")

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}

        gamma_node = node.get_input_node(node.inputs[1])
        if not isinstance(gamma_node, Constant):
            raise TypeError("Only consant gammas supported")
        gamma = gamma_node.value
        attributes['gamma_data'] = gamma
        node.inputs[1] = ''
        model.remove_node(gamma_node, rewire=False)

        beta_node = node.get_input_node(node.inputs[2])
        if not isinstance(beta_node, Constant):
            raise TypeError("Only consant betas supported")
        beta = beta_node.value
        attributes['beta_data'] = beta
        node.inputs[2] = ''
        model.remove_node(beta_node, rewire=False)

        moving_mean_node = node.get_input_node(node.inputs[3])
        if not isinstance(moving_mean_node, Constant):
            raise TypeError("Only consant moving_means supported")
        moving_mean = moving_mean_node.value
        attributes['mean_data'] = moving_mean
        node.inputs[3] = ''
        model.remove_node(moving_mean_node, rewire=False)

        moving_variance_node = node.get_input_node(node.inputs[4])
        if not isinstance(moving_variance_node, Constant):
            raise TypeError("Only consant moving_variances supported")
        moving_variance = moving_variance_node.value
        attributes['variance_data'] = moving_variance
        node.inputs[4] = ''
        model.remove_node(moving_variance_node, rewire=False)

        # scale = gamma / np.sqrt(moving_variance + node.get_attr('epsilon'))
        # bias = beta - gamma * moving_mean / np.sqrt(moving_variance + node.get_attr('epsilon'))
        # attributes["scale_data"] = scale
        # attributes["bias_data"] = bias

        new_node = model.make_node(BatchNormalization, node.name, attributes, [node.inputs[0]], [x for x in node.outputs])

        model.replace_node(node, new_node)

        return True


class ConstantBatchNormFusion(OptimizerPass):
    """
    Merge BatchNorm into Const (after parameters have already been merged in BatchNormalization)
    """

    def match(self, node):
        is_match = (
            isinstance(node, BatchNormalization)
            and not any(node.inputs[1:])
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and not node.get_input_node(node.inputs[0]).get_attr("quant_precision")
        )
        return is_match

    def transform(self, model, node):
        """
        Remove the batch norm
        """
        const_node = node.get_input_node(node.inputs[0])

        new_val = const_node.value * node.weights["scale"].data_unquantized + node.weights["bias"].data_unquantized
        const_node.set_attr("value", new_val)
        const_node.set_attr("quantizer", node.get_attr("quantizer"))  # None if not defined
        const_node.set_attr("quant_precision", node.get_attr("quant_precision"))

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
        prev_node = node.get_input_node(node.inputs[0])
        basic_match = (
            isinstance(node, BatchNormalization)
            and isinstance(prev_node, BatchNormalization)
            and not prev_node.get_attr("quant_precision")
        )

        # check for compatibility to merge
        if basic_match:
            s0 = prev_node.weights['scale'].data_unquantized
            b0 = prev_node.weights['bias'].data_unquantized
            s1 = node.weights['scale'].data_unquantized
            b1 = node.weights['bias'].data_unquantized
            scale_compatible = (
                (prev_node.get_attr("scale_quantizer") is None and node.get_attr("scale_quantizer") is None)
                or (s0 == np.ones_like(s0)).all()
                or (s1 == np.ones_like(s1)).all()
            )
            bias_compatible = (
                (prev_node.get_attr("bias_quantizer") is None and node.get_attr("bias_quantizer") is None)
                or (b0 == np.zeros_like(b0)).all()
                or (b1 == np.zeros_like(b1)).all()
            )
            return scale_compatible and bias_compatible
        else:
            return False

    def transform(self, model, node):
        prev_node = node.get_input_node(node.inputs[0])

        s0 = prev_node.weights['scale'].data_unquantized
        b0 = prev_node.weights['bias'].data_unquantized
        s1 = node.weights['scale'].data_unquantized
        b1 = node.weights['bias'].data_unquantized

        s_quantizer = (
            node.get_attr("scale_quantizer") if (s0 == np.ones_like(s0)).all() else prev_node.get_attr("scale_quantizer")
        )
        b_quantizer = (
            node.get_attr("bias_quantizer") if (b0 == np.zeros_like(b0)).all() else prev_node.get_attr("bias_quantizer")
        )

        node.set_attr("scale_quantizer", s_quantizer)
        node.set_attr("bias_quantizer", b_quantizer)
        if s_quantizer:
            node.set_attr("scale_precision", s_quantizer.hls_type)
        if b_quantizer:
            node.set_attr("bias_precision", b_quantizer.hls_type)

        scale_new = s0 * s1
        bias_new = s1 * b0 + b1

        # call function so that quantizer would be called if needed
        node.add_weights_variable(name='scale', var_name='s{index}', data=scale_new)
        node.add_weights_variable(name='bias', var_name='b{index}', data=bias_new)

        model.remove_node(prev_node, rewire=True)
        return True
