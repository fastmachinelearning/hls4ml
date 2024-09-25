'''
This file includes optimizations related to moving the ApplyAphas across MatMul and Conv nodes.

TODO:  Check that biases are properly handled. (Attempt to do it via Merge)

'''

import numpy as np

from hls4ml.model.layers import ApplyAlpha, Constant, Conv, MatMul, Merge
from hls4ml.model.optimizer import OptimizerPass


class ScaleDownMatMul(OptimizerPass):
    '''Shift an ApplyAlpha below a MatMul'''

    def match(self, node):
        '''
        Check to see if we have a MatMul with at least one input ApplyAlpha.
        Note, if both are this optimizer runs twice.
        '''
        is_match = (
            isinstance(node, MatMul)
            and len(node.inputs) == 2
            and (
                isinstance(node.get_input_node(node.inputs[0]), ApplyAlpha)
                or isinstance(node.get_input_node(node.inputs[1]), ApplyAlpha)
            )
        )
        return is_match

    def transform(self, model, node):
        # determine input with ApplyAlpha. If both, first propagate apply alpha associated with a constant
        is_aa = [False, False]
        from_const = [False, False]
        inp = [node.get_input_node(node.inputs[0]), node.get_input_node(node.inputs[1])]
        for i in range(2):
            if isinstance(inp[i], ApplyAlpha):
                is_aa[i] = True
                from_const[i] = isinstance(inp[i].get_input_node(inp[i].inputs[0]), Constant)

        # prefer alpha from constant
        if from_const[0]:
            alpha_idx = 0
        elif from_const[1]:
            alpha_idx = 1
        elif is_aa[0]:
            alpha_idx = 0
        else:
            alpha_idx = 1  # is_aa[1] must be true

        apply_alpha = inp[alpha_idx]
        other_idx = 0 if alpha_idx else 1

        # Check if we can move
        scale = apply_alpha.weights['scale'].data_unquantized
        bias = apply_alpha.weights['bias'].data_unquantized

        scale1d = np.ravel(scale)
        if (scale1d[0] == scale).all():
            # scalar scale
            scale = np.array(scale1d[0])

        bias1d = np.ravel(bias)
        if (bias1d[0] == bias).all():
            # scalar bias
            bias = np.array(bias1d[0])

        output = node.get_output_variable()
        # to remove warning, since these get set again
        new_attrs = {k: v for k, v in apply_alpha.attributes.items() if k not in ('trace', 'precision')}

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                newscale = np.broadcast_to(scale, output.shape)  # check size compatibility
                newbias = np.zeros(output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False

        # if did not succeed in propagating, try again
        if not can_propagate and isinstance(inp[other_idx], Constant):
            # can handle nonzero bias in some cases if other value is a Constant
            try:
                newscale = np.broadcast_to(scale, output.shape)  # check size compatibility
                newbias = np.broadcast_to(inp[other_idx].attributes['value'] * bias, output.shape)
                new_attrs.pop('bias_precision', None)  # remove special bias precision settings
                can_propagate = True
            except ValueError:
                can_propagate = False

        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_attrs['scale_data'] = newscale
        new_attrs['bias_data'] = newbias

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


class ScaleDownAdd(OptimizerPass):
    '''Shift an identical ApplyAlpha below a Merge (Add)'''

    def match(self, node):
        '''Check to see if we have an add with two ApplyAlphas with identical scale'''
        is_match = isinstance(node, Merge) and len(node.inputs) == 2 and node.attributes["op"] == "add"
        if is_match:
            in0 = node.get_input_node(node.inputs[0])
            in1 = node.get_input_node(node.inputs[1])
            is_match = (
                isinstance(in0, ApplyAlpha)
                and isinstance(in1, ApplyAlpha)
                and (in0.weights['scale'].data_unquantized == in1.weights['scale'].data_unquantized).all()
            )
        return is_match

    def transform(self, model, node):
        in0 = node.get_input_node(node.inputs[0])
        in1 = node.get_input_node(node.inputs[1])

        # Check if we can move
        scale = in0.weights['scale'].data_unquantized
        bias0 = in0.weights['bias'].data_unquantized
        bias1 = in1.weights['bias'].data_unquantized
        try:
            bias = bias0 + bias1
        except ValueError:
            return False

        model.remove_node(in0)
        model.remove_node(in1)

        new_attrs = in0.attributes
        new_attrs['scale_data'] = scale
        new_attrs['bias_data'] = bias

        new_node = model.make_node('ApplyAlpha', in0.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


class ScaleDownConv(OptimizerPass):
    '''Shift an ApplyAlpha on input below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = isinstance(node, Conv) and isinstance(node.get_input_node(node.inputs[0]), ApplyAlpha)

        return is_match

    def transform(self, model, node):
        apply_alpha = node.get_input_node(node.inputs[0])

        # Check if we can move
        scale = apply_alpha.weights['scale'].data_unquantized
        bias = apply_alpha.weights['bias'].data_unquantized

        scale1d = np.ravel(scale)
        if (scale1d[0] == scale).all():
            # scalar scale
            scale = np.array(scale1d[0])

        bias1d = np.ravel(bias)
        if (bias1d[0] == bias).all():
            # scalar bias
            bias = np.array(bias1d[0])

        output = node.get_output_variable()
        # to remove warning, since these get set again
        new_attrs = {k: v for k, v in apply_alpha.attributes.items() if k not in ('trace', 'precision')}

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                newscale = np.broadcast_to(scale, output.shape)  # check broadcastable
                newbias = np.zeros(output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False

        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_attrs['scale_data'] = newscale
        new_attrs['bias_data'] = newbias

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


class ScaleDownWeightConv(OptimizerPass):
    '''Shift an ApplyAlpha weight (from conv side) below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (
            isinstance(node, Conv) and len(node.inputs) > 1 and isinstance(node.get_input_node(node.inputs[1]), ApplyAlpha)
        )

        return is_match

    def transform(self, model, node):
        apply_alpha = node.get_input_node(node.inputs[1])

        # Check if we can move
        scale = apply_alpha.weights['scale'].data_unquantized
        bias = apply_alpha.weights['bias'].data_unquantized

        scale1d = np.ravel(scale)
        if (scale1d[0] == scale).all():
            # scalar scale
            scale = np.array(scale1d[0])

        bias1d = np.ravel(bias)
        if (bias1d[0] == bias).all():
            # scalar bias
            bias = np.array(bias1d[0])

        output = node.get_output_variable()
        # to remove warning, since these get set again
        new_attrs = {k: v for k, v in apply_alpha.attributes.items() if k not in ('trace', 'precision')}

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                if scale.ndim > 1:
                    # undo any broadcast_to
                    reduced_scale = _remove_redundant_dims(scale)
                    if reduced_scale.shape[-1] == 1:
                        reduced_scale = reduced_scale[..., 0]
                        if node.attributes['n_dim'] == 1:
                            scale_trans = np.transpose(reduced_scale, (1, 0))
                        else:
                            scale_trans = np.transpose(reduced_scale, (1, 2, 0))
                        newscale = np.broadcast_to(scale_trans, output.shape)  # make sure broadcastable
                        can_propagate = True
                else:
                    newscale = np.broadcast_to(scale, output.shape)  # make sure broadcastable
                    can_propagate = True
                newbias = np.zeros(output.shape)
            except ValueError:
                can_propagate = False

        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_attrs['scale_data'] = newscale
        new_attrs['bias_data'] = newbias

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


class ScaleDownBiasConv(OptimizerPass):
    '''Shift an ApplyAlpha bias (from conv side) below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (
            isinstance(node, Conv) and len(node.inputs) > 2 and isinstance(node.get_input_node(node.inputs[2]), ApplyAlpha)
        )

        return is_match

    def transform(self, model, node):
        apply_alpha = node.get_input_node(node.inputs[2])

        # Check if we can move
        scale = apply_alpha.weights['scale'].data_unquantized
        bias = apply_alpha.weights['bias'].data_unquantized

        scale1d = np.ravel(scale)
        if (scale1d[0] == scale).all():
            # scalar scale
            scale = np.array(scale1d[0])

        bias1d = np.ravel(bias)
        if (bias1d[0] == bias).all():
            # scalar bias
            bias = np.array(bias1d[0])

        output = node.get_output_variable()
        # to remove warning, since these get set again
        new_attrs = {k: v for k, v in apply_alpha.attributes.items() if k not in ('trace', 'precision')}

        can_propagate = False
        if not scale.shape and scale == 1:
            # No scale, just additional bias
            try:
                newscale = np.ones(output.shape)
                newbias = np.broadcast_to(bias, output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False

        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_attrs['scale_data'] = newscale
        new_attrs['bias_data'] = newbias

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


def _remove_redundant_dims(X):
    """This is somewhat of the inverse of broadcast-to. It sets the dimension size to 1 if all values are identical"""

    shape = X.shape
    for i in range(len(shape)):
        reduced = np.expand_dims(np.take(X, 0, axis=i), axis=i)
        if np.all(reduced == X):
            X = reduced
    return X
