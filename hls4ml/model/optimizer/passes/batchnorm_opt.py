import warnings

import numpy as np

from hls4ml.model.layers import BatchNormalization, BatchNormOnnx, Constant
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import QuantNodeQuantizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, UnspecifiedPrecisionType

_base_attributes = ('epsilon', 'n_in', 'n_filt')


class BatchNormOnnxConstantParameters(OptimizerPass):
    """Remove Constant from the BatchNormalization node parameters (but not input[0])"""

    def match(self, node):
        is_match = isinstance(node, BatchNormOnnx) and any(node.inputs[1:])

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from the BatchNormalization node parameters (but not input[0])

        TODO:  Currently the quantizers are not actually used by the underlying layer.
        """

        if not (len(node.inputs) == 5 and all(node.inputs)):
            raise ValueError('All 5 BatchNormOnnnx inputs need to be defined')

        attributes = {k: node.attributes[k] for k in _base_attributes if k in node.attributes}

        gamma_node = node.get_input_node(node.inputs[1])
        if not isinstance(gamma_node, Constant):
            raise TypeError('Only constant gammas supported')
        gamma = gamma_node.attributes['value']
        attributes['gamma_data'] = gamma
        attributes['gamma_quantizer'] = gamma_node.get_attr('quantizer')

        node.inputs[1] = ''
        model.remove_node(gamma_node, rewire=False)

        beta_node = node.get_input_node(node.inputs[2])
        if not isinstance(beta_node, Constant):
            raise TypeError('Only constant betas supported')
        beta = beta_node.attributes['value']
        attributes['beta_data'] = beta
        attributes['beta_quantizer'] = beta_node.get_attr('quantizer')
        node.inputs[2] = ''
        model.remove_node(beta_node, rewire=False)

        moving_mean_node = node.get_input_node(node.inputs[3])
        if not isinstance(moving_mean_node, Constant):
            raise TypeError('Only constant moving_means supported')
        moving_mean = moving_mean_node.attributes['value']
        attributes['mean_data'] = moving_mean
        attributes['mean_quantizer'] = moving_mean_node.get_attr('quantizer')
        node.inputs[3] = ''
        model.remove_node(moving_mean_node, rewire=False)

        moving_variance_node = node.get_input_node(node.inputs[4])
        if not isinstance(moving_variance_node, Constant):
            raise TypeError('Only constant moving_variances supported')
        moving_variance = moving_variance_node.attributes['value']
        attributes['variance_data'] = moving_variance
        attributes['variance_quantizer'] = moving_variance_node.get_attr('quantizer')
        node.inputs[4] = ''
        model.remove_node(moving_variance_node, rewire=False)

        node.inputs = [inp for inp in node.inputs if inp]
        if len(node.inputs) != 1:
            raise RuntimeError('The QONNX batchnorm had unexpected inputs.')

        new_node = model.make_node(BatchNormalization, node.name, attributes, [node.inputs[0]], [x for x in node.outputs])

        model.replace_node(node, new_node)

        return True


# Most likely this case is removed by qonnx cleaning
class ConstantBatchNormFusion(OptimizerPass):
    """
    Merge BatchNorm into Const (after parameters have already been merged in BatchNormalization)
    """

    def match(self, node):
        is_match = (
            isinstance(node, BatchNormalization)
            and not any(node.inputs[1:])
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and isinstance(
                node.get_input_node(node.inputs[0]).get_output_variable().type.precision, UnspecifiedPrecisionType
            )
        )
        return is_match

    def transform(self, model, node):
        """
        Remove the batch norm
        """
        warnings.warn('ConstantBatchNormFusion should probably not be triggered. Check the optimizer order.', stacklevel=2)
        const_node = node.get_input_node(node.inputs[0])

        const_prec = const_node.get_output_variable().type.precision

        new_val = (
            const_node.attributes['value'] * node.weights['scale'].data_unquantized + node.weights['bias'].data_unquantized
        )

        const_node.set_attr('value', new_val)
        const_node.set_attr('quantizer', node.get_attr('quantizer'))  # None if not defined

        if isinstance(node.get_output_variable().type.precision, UnspecifiedPrecisionType):
            if isinstance(const_prec, UnspecifiedPrecisionType):
                pass  # leave it as is
            else:
                const_node.get_output_variable().type.precision = UnspecifiedPrecisionType()  # default
                # propagate precision
                scale_q = node.get_attr('scale_quantizer')
                bias_q = node.get_attr('bias_quantizer')
                if scale_q and bias_q:
                    # propagate precsion
                    scale_prec = scale_q.hls_type
                    bias_prec = bias_q.hls_type
                    if scale_prec not in (IntegerPrecisionType, FixedPrecisionType) or bias_prec not in (
                        IntegerPrecisionType,
                        FixedPrecisionType,
                    ):
                        print("Warning:  output type not propagated for constant merge")
                    else:
                        signed_prod = const_prec.signed or scale_prec.signed
                        w_prod = const_prec.width + scale_prec.width
                        i_prod = const_prec.integer + scale_prec.integer
                        signed = signed_prod or bias_prec.signed
                        i_tot = (
                            max(
                                i_prod + (bias_prec.signed and not signed_prod),
                                bias_prec.ingeter + (signed_prod and not bias_prec.signed),
                            )
                            + 1
                        )
                        w_tot = i_tot + max(w_prod - i_prod, bias_prec.width - bias_prec.integer)
                        new_prec = FixedPrecisionType(w_tot, i_tot, signed)
                        const_node.set_attr('quantizer', QuantNodeQuantizer(new_prec))
                        const_node.get_output_variable().type.precision = new_prec
        else:
            const_node.get_output_variable().type.precision = node.get_output_variable().type.precision

        # remove the batch norm node
        model.remove_node(node, rewire=True)

        return True


class FuseConsecutiveBatchNormalization(OptimizerPass):
    """
    OptimizerPass to merge consecutive BatchNormalization layers, only if the earlier one does not have the output type
    specified. There is a further check on the compatibility to merge: except in cases when merging a scale of 1 or a
    bias of 0, this does not merge when both scales or both biases are quantized.

    Note:  Consider restricting this to ApplyAlpha.  Batch Normalization-style quantization seems to be ignored.

    Note:  This optimizer may not be safe if weights are updateable, in particular if a scale can go from ones to other
    values or if a bias can go from zeros to other values.
    """

    def match(self, node):
        prev_node = node.get_input_node()
        basic_match = (
            isinstance(node, BatchNormalization)
            and isinstance(prev_node, BatchNormalization)
            and isinstance(prev_node.get_output_variable().type.precision, UnspecifiedPrecisionType)
        )

        # check for compatibility to merge
        if basic_match:
            s0 = prev_node.weights['scale'].data_unquantized
            b0 = prev_node.weights['bias'].data_unquantized
            s1 = node.weights['scale'].data_unquantized
            b1 = node.weights['bias'].data_unquantized
            scale_compatible = (
                (prev_node.get_attr('scale_quantizer') is None or node.get_attr('scale_quantizer') is None)
                or (s0 == np.ones_like(s0)).all()
                or (s1 == np.ones_like(s1)).all()
            )
            bias_compatible = (
                (prev_node.get_attr('bias_quantizer') is None or node.get_attr('bias_quantizer') is None)
                or (b0 == np.zeros_like(b0)).all()
                or (b1 == np.zeros_like(b1)).all()
            )
            return scale_compatible and bias_compatible
        else:
            return False

    def transform(self, model, node):
        prev_node = node.get_input_node()

        prev_map = prev_node.get_output_use_map()
        if len(prev_map[prev_node.outputs[0]]) > 1:
            return False

        s0 = prev_node.weights['scale'].data_unquantized
        b0 = prev_node.weights['bias'].data_unquantized
        s1 = node.weights['scale'].data_unquantized
        b1 = node.weights['bias'].data_unquantized

        if (s0 == np.ones_like(s0)).all():
            s_quantizer = node.get_attr('scale_quantizer')
        elif (s1 == np.ones_like(s1)).all():
            s_quantizer = prev_node.get_attr('scale_quantizer')
        else:
            s_quantizer = None

        if (b0 == np.ones_like(b0)).all():
            b_quantizer = node.get_attr('bias_quantizer')
        elif (b1 == np.ones_like(b1)).all():
            b_quantizer = prev_node.get_attr('bias_quantizer')
        else:
            b_quantizer = None

        node.set_attr('scale_quantizer', s_quantizer)
        node.set_attr('bias_quantizer', b_quantizer)

        scale_new = s0 * s1
        bias_new = s1 * b0 + b1

        # Not sure if this setting of this is useful
        s_prec = None
        if s_quantizer is None and (scale_new == np.ones_like(scale_new)).all():
            if (
                isinstance(prev_node.weights['scale'].type, IntegerPrecisionType)
                and isinstance(node.weights['scale'].type, IntegerPrecisionType)
                and prev_node.weights['scale'].type.width == 1
                and node.weights['scale'].type.width == 1
            ):
                s_prec = node.weights['scale'].type

        b_prec = None
        if b_quantizer is None and (bias_new == np.zeros_like(bias_new)).all():
            if (
                isinstance(prev_node.weights['bias'].type, IntegerPrecisionType)
                and isinstance(node.weights['bias'].type, IntegerPrecisionType)
                and prev_node.weights['bias'].type.width == 1
                and node.weights['bias'].type.width == 1
            ):
                b_prec = node.weights['bias'].type

        # call function so that quantizer would be called if needed
        node.add_weights_variable(name='scale', var_name='s{index}', data=scale_new, quantizer=s_quantizer, precision=s_prec)
        node.add_weights_variable(name='bias', var_name='b{index}', data=bias_new, quantizer=b_quantizer, precision=b_prec)

        model.remove_node(prev_node, rewire=True)
        return True


class RemoveNopBatchNormalization(OptimizerPass):
    """
    OptimizerPass to remove batch normalizations that do nothing (scale 1, bias 0)

    Note:  This optimizer may not be safe if weights are updateable.
    """

    def match(self, node):
        if isinstance(node, BatchNormalization):
            s0 = node.weights['scale'].data_unquantized
            b0 = node.weights['bias'].data_unquantized
            return (s0 == np.ones_like(s0)).all() and (b0 == np.zeros_like(b0)).all()
        else:
            return False

    def transform(self, model, node):
        model.remove_node(node, rewire=True)
        return True
