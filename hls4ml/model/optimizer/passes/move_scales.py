'''
This file includes optimizations related to moving the ApplyAphas across MatMul and Conv nodes.

TODO:  Check that biases are properly handled. (Attempt to do it via Merge)

'''
from copy import deepcopy
import numpy as np
from hls4ml.model.hls_layers import ApplyAlpha, Constant, MatMul, Conv, Merge
from hls4ml.converters.onnx.quantizer import QuantNodeQuantizer
from hls4ml.model.optimizer import OptimizerPass

class ApplyAlphaDownMatMul(OptimizerPass):
    '''Shift an ApplyAlpha below a MatMul'''

    def match(self, node):
        '''Check to see if we have a MatMul with at least one input ApplyAlpha. Note, if both are this optimition runs twice'''
        is_match = (isinstance(node, MatMul) and len(node.inputs) == 2
                    and (isinstance(node.get_input_node(node.inputs[0]), ApplyAlpha)
                         or isinstance(node.get_input_node(node.inputs[1]), ApplyAlpha)))
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

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                newscale = np.broadcast_to(scale, output.shape)
                newbias = np.zeros(output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False

        # if did not succeed in propagating, try again
        if not can_propagate and isinstance(inp[other_idx], Constant):
            # can handle nonzero bias in some cases if other value is a Constant
            try:
                newscale = np.broadcast_to(scale, output.shape)
                newbias = np.broadcast_to(inp[other_idx].value * bias, output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False
    
        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, apply_alpha.attributes, [x for x in node.outputs])
        new_node.add_weights(newscale)
        new_node.add_bias(newbias)
        model.insert_node(new_node)
        return True

class ApplyAlphaDownConv(OptimizerPass):
    '''Shift an ApplyAlpha on input below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (isinstance(node, Conv)
                    and isinstance(node.get_input_node(node.inputs[0]), ApplyAlpha))

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

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                newscale = np.broadcast_to(scale, output.shape)
                newbias = np.zeros(output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False
   
        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, apply_alpha.attributes, [x for x in node.outputs])
        new_node.add_weights(newscale)
        new_node.add_bias(newbias)
        model.insert_node(new_node)
        return True

class ApplyAlphaWeightDownConv(OptimizerPass):
    '''Shift an ApplyAlpha weight (from conv side) below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (isinstance(node, Conv) and len(node.inputs) > 1
                    and isinstance(node.get_input_node(node.inputs[1]), ApplyAlpha))

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

        can_propagate = False
        if not bias.shape and bias == 0:
            # zero bias, propagate through, if possible
            # (always possible if scale is scalar)
            try:
                newscale = np.broadcast_to(scale, output.shape)
                newbias = np.zeros(output.shape)
                can_propagate = True
            except ValueError:
                can_propagate = False
   
        if not can_propagate:
            return False

        model.remove_node(apply_alpha)

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, apply_alpha.attributes, [x for x in node.outputs])
        new_node.add_weights(newscale)
        new_node.add_bias(newbias)
        model.insert_node(new_node)
        return True

class ApplyAlphaBiasDownConv(OptimizerPass):
    '''Shift an ApplyAlpha bias (from conv side) below a Conv'''

    def match(self, node):
        '''Shift an ApplyAlpha from the Weight'''
        is_match = (isinstance(node, Conv) and len(node.inputs) > 2
                    and isinstance(node.get_input_node(node.inputs[2]), ApplyAlpha))

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

        new_node = model.make_node('ApplyAlpha', apply_alpha.name, apply_alpha.attributes, [x for x in node.outputs])
        new_node.add_weights(newscale)
        new_node.add_bias(newbias)
        model.insert_node(new_node)
        return True
