'''
This file includes optimizations related to moving the ApplyAphas across MatMul and Conv nodes.

TODO:  Check that biases are properly handled. (Attempt to do it via Merge)

'''

import warnings

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

        scale, bias = _make_scalar(scale, bias)

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
            warnings.warn(
                'Failed to propagate quantization scales down MatMul node; model probably not suppored.', stacklevel=1
            )
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
            warnings.warn(
                'Failed to propagate quantization scales down Add node; model probably not suppored.', stacklevel=1
            )
            return False

        model.remove_node(in0)
        model.remove_node(in1)

        new_attrs = in0.attributes
        new_attrs['scale_data'] = scale
        new_attrs['bias_data'] = bias

        new_node = model.make_node('ApplyAlpha', in0.name, new_attrs, [x for x in node.outputs])
        model.insert_node(new_node)
        return True


class BiasDownAdd(OptimizerPass):
    '''Shift a ApplyAlpha with only bias below a Merge (Add)'''

    def match(self, node):
        '''Match if there is only one ApplyAlpha. If there are two, if the scale of both is 0, they would
        match the ScaleDownAdd, so this optimizer does not need to handle that case.
        '''
        is_match = isinstance(node, Merge) and len(node.inputs) == 2 and node.attributes["op"] == "add"
        if is_match:
            in0 = node.get_input_node(node.inputs[0])
            in1 = node.get_input_node(node.inputs[1])
            is_match = (isinstance(in0, ApplyAlpha) or isinstance(in1, ApplyAlpha)) and not (
                isinstance(in0, ApplyAlpha) and isinstance(in1, ApplyAlpha)
            )  # only one ApplyAlpha
        return is_match

    def transform(self, model, node):
        in0 = node.get_input_node(node.inputs[0])
        in1 = node.get_input_node(node.inputs[1])

        alpha_node = in0 if isinstance(in0, ApplyAlpha) else in1

        # Check if we can move
        scale = alpha_node.weights['scale'].data_unquantized

        if (scale == 0).all():
            model.remove_node(alpha_node)
            new_node = model.make_node('ApplyAlpha', alpha_node.name, alpha_node.attributes, [x for x in node.outputs])
            model.insert_node(new_node)
            return True
        else:
            warnings.warn('Failed to propagate quantization bias down Add node; model probably not suppored.', stacklevel=1)
            return False


class ScaleDownConv(OptimizerPass):
    '''Shift an ApplyAlpha on a Conv with 2-3 inputs'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (
            isinstance(node, Conv)
            and len(node.inputs) > 1
            and (
                isinstance(node.get_input_node(node.inputs[0]), ApplyAlpha)
                or isinstance(node.get_input_node(node.inputs[1]), ApplyAlpha)
                or (len(node.inputs) == 3 and isinstance(node.get_input_node(node.inputs[2]), ApplyAlpha))
            )
        )
        return is_match

    def transform(self, model, node):
        in0 = node.get_input_node(node.inputs[0])
        in1 = node.get_input_node(node.inputs[1])
        in2 = node.get_input_node(node.inputs[2]) if len(node.inputs) == 3 else None

        aa0 = isinstance(in0, ApplyAlpha)
        aa1 = isinstance(in1, ApplyAlpha)
        aa2 = isinstance(in2, ApplyAlpha) if len(node.inputs) == 3 else False

        if not isinstance(in1, (Constant, ApplyAlpha)):
            raise RuntimeError("The weight node needs to be ApplyAlpha or Constant")
        if len(node.inputs) == 3 and not isinstance(in2, (Constant, ApplyAlpha)):
            raise RuntimeError("The bias node needs to be ApplyAlpha or Constant")

        scale0 = in0.weights['scale'].data_unquantized if aa0 else None
        bias0 = in0.weights['bias'].data_unquantized if aa0 else None
        scale1 = in1.weights['scale'].data_unquantized if aa1 else None
        bias1 = in1.weights['bias'].data_unquantized if aa1 else None
        scale2 = in2.weights['scale'].data_unquantized if aa2 else None
        bias2 = in2.weights['bias'].data_unquantized if aa2 else None

        # If possible, make scale and bias have scalar values
        if aa0:
            scale0, bias0 = _make_scalar(scale0, bias0)
        if aa1:
            scale1, bias1 = _make_scalar(scale1, bias1)
        if aa2:
            scale2, bias2 = _make_scalar(scale2, bias2)

        output = node.get_output_variable()
        if aa0 and not aa1 and not aa2:
            # only datapath has a scale
            bias = in2.attributes['value'] if len(node.inputs) == 3 else 0
            conv_nobias = np.all(bias == 0)

            can_propagate = False
            if not bias0.shape and bias0 == 0:
                # No zero offset, propagate through, if possible
                # (always possible if scale is scalar)
                if conv_nobias:
                    try:
                        newscale = np.broadcast_to(_remove_redundant_dims(scale0), output.shape)  # check broadcastable
                        newbias = np.zeros(output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
                elif not scale0.shape:
                    # scalar scale0
                    try:
                        newscale = np.broadcast_to(scale0, output.shape)  # check broadcastable
                        newbias = np.broadcast_to(bias * (1 - scale0), output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in0.attributes.items() if k not in ('trace', 'precision')}
            new_name = in0.name
            model.remove_node(in0)

        elif not aa0 and aa1 and not aa2:
            # only weights have an ApplyAlpha
            bias = in2.attributes['value'] if len(node.inputs) == 3 else 0
            conv_nobias = np.all(bias == 0)

            can_propagate = False
            if not bias1.shape and bias1 == 0:
                # No zero offset, propagate through, if possible
                # (always possible if scale is scalar)
                if conv_nobias:
                    try:
                        if scale1.ndim > 1:
                            # undo any broadcast_to
                            reduced_scale = _remove_redundant_dims(scale1)
                            if reduced_scale.shape[-1] == 1:
                                reduced_scale = reduced_scale[..., 0]
                                if node.attributes['n_dim'] == 1:
                                    scale_trans = np.transpose(reduced_scale, (1, 0))
                                else:
                                    scale_trans = np.transpose(reduced_scale, (1, 2, 0))
                                newscale = np.broadcast_to(scale_trans, output.shape)  # make sure broadcastable
                                can_propagate = True
                        else:
                            newscale = np.broadcast_to(scale1, output.shape)  # make sure broadcastable
                            can_propagate = True
                        newbias = np.zeros(output.shape)
                    except ValueError:
                        can_propagate = False
                elif not scale1.shape:
                    # scalar scale1
                    try:
                        newscale = np.broadcast_to(scale1, output.shape)  # check broadcastable
                        newbias = np.broadcast_to(bias * (1 - scale1), output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in0.attributes.items() if k not in ('trace', 'precision')}
            new_name = in1.name
            model.remove_node(in1)

        elif not aa0 and not aa1 and aa2:
            # only bias has a scale

            can_propagate = False
            if not scale2.shape and scale2 == 1:
                # No scale, just additional bias
                try:
                    newscale = np.ones(output.shape)
                    newbias = np.broadcast_to(bias2, output.shape)
                    can_propagate = True
                except ValueError:
                    can_propagate = False

            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in2.attributes.items() if k not in ('trace', 'precision')}
            new_name = in2.name
            model.remove_node(in2)

        elif aa0 and aa1 and not aa2:
            # dataflow and weights have an ApplyAlpha
            bias = in2.attributes['value'] if len(node.inputs) == 3 else 0
            conv_nobias = np.all(bias == 0)

            can_propagate = False
            if not bias0.shape and bias0 == 0 and not bias1.shape and bias1 == 0:
                # No zero offset, propagate through, if possible
                # (always possible if scale is scalar)
                if conv_nobias:
                    try:
                        if scale1.ndim > 1:
                            # undo any broadcast_to
                            reduced_scale0 = _remove_redundant_dims(scale0) if scale0.ndim > 1 else scale0
                            reduced_scale1 = _remove_redundant_dims(scale1)
                            reduced_scale = reduced_scale0 @ reduced_scale1
                            if reduced_scale.shape[-1] == 1:
                                reduced_scale = reduced_scale[..., 0]
                                if node.attributes['n_dim'] == 1:
                                    scale_trans = np.transpose(reduced_scale, (1, 0))
                                else:
                                    scale_trans = np.transpose(reduced_scale, (1, 2, 0))
                                newscale = np.broadcast_to(scale_trans, output.shape)  # make sure broadcastable
                                can_propagate = True
                        elif scale0.ndim > 1:
                            # scale1 is scalar
                            # undo any broadcast_to
                            reduced_scale0 = _remove_redundant_dims(scale0)
                            reduced_scale = scale1 * reduced_scale0
                            if reduced_scale.shape[-1] == 1:
                                reduced_scale = reduced_scale[..., 0]
                                if node.attributes['n_dim'] == 1:
                                    scale_trans = np.transpose(reduced_scale, (1, 0))
                                else:
                                    scale_trans = np.transpose(reduced_scale, (1, 2, 0))
                                newscale = np.broadcast_to(scale_trans, output.shape)  # make sure broadcastable
                                can_propagate = True
                        else:
                            newscale = np.broadcast_to(scale0 * scale1, output.shape)  # make sure broadcastable
                            can_propagate = True
                        newbias = np.zeros(output.shape)
                    except ValueError:
                        can_propagate = False
                elif not scale0.shape and not scale1.shape:
                    # scalar scale1
                    try:
                        newscale = np.broadcast_to(scale0 * scale1, output.shape)  # check broadcastable
                        newbias = np.broadcast_to(bias * (1 - scale0 * scale1), output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in0.attributes.items() if k not in ('trace', 'precision')}
            new_name = in1.name
            model.remove_node(in0)
            model.remove_node(in1)

        elif aa0 and not aa1 and aa2:
            # datapath and bias have a scale

            can_propagate = False
            if not bias0.shape and bias0 == 0 and not scale2.shape and not scale0.shape and scale2 == scale0:
                # scalar scale0, no bais0 and scale2.
                try:
                    newscale = np.broadcast_to(scale0, output.shape)  # check broadcastable
                    newbias = np.broadcast_to(bias2, output.shape)
                    can_propagate = True
                except ValueError:
                    can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in0.attributes.items() if k not in ('trace', 'precision')}
            new_name = in0.name
            model.remove_node(in0)
            model.remove_node(in2)

        elif not aa0 and aa1 and aa2:
            # only weights and bias have an ApplyAlpha

            can_propagate = False
            if not bias1.shape and bias1 == 0 and not scale2.shape and not scale1.shape and scale2 == scale1:
                # No zero offset, propagate through, if possible
                # (always possible if scale is scalar)
                if not scale1.shape:
                    # scalar scale1
                    try:
                        newscale = np.broadcast_to(scale1, output.shape)  # check broadcastable
                        newbias = np.broadcast_to(bias2, output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in1.attributes.items() if k not in ('trace', 'precision')}
            new_name = in1.name
            model.remove_node(in1)
            model.remove_node(in2)

        elif aa0 and aa1 and aa2:
            # have all

            can_propagate = False
            if (
                not bias0.shape
                and bias0 == 0
                and not bias1.shape
                and bias1 == 0
                and not scale2.shape
                and not scale1.shape
                and not scale0.shape
                and scale2 == scale1 * scale0
            ):
                # No zero offset, propagate through, if possible
                # (always possible if scale is scalar)
                if not scale1.shape:
                    # scalar scale1
                    try:
                        newscale = np.broadcast_to(scale0 * scale1, output.shape)  # check broadcastable
                        newbias = np.broadcast_to(bias2, output.shape)
                        can_propagate = True
                    except ValueError:
                        can_propagate = False
            if not can_propagate:
                warnings.warn(
                    'Failed to propagate quantization scales down Conv node; model probably not suppored.', stacklevel=1
                )
                return False

            # to remove warning, since these get set again
            new_attrs = {k: v for k, v in in0.attributes.items() if k not in ('trace', 'precision')}
            new_name = in0.name
            model.remove_node(in0)
            model.remove_node(in1)
            model.remove_node(in2)

        # after the big if-else above
        new_attrs['scale_data'] = newscale
        new_attrs['bias_data'] = newbias

        new_node = model.make_node('ApplyAlpha', new_name, new_attrs, [x for x in node.outputs])
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


def _make_scalar(scale, bias):
    """Make the scale and bias scalar if possible"""
    scale1d = np.ravel(scale)
    if (scale1d[0] == scale).all():
        # scalar scale
        scale = np.array(scale1d[0])

    bias1d = np.ravel(bias)
    if (bias1d[0] == bias).all():
        # scalar bias
        bias = np.array(bias1d[0])

    return scale, bias
