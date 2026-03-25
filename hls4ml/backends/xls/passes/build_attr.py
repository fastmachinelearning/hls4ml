# Typing imports
from __future__ import annotations  # makes all annotations into strings

from typing import Literal, Optional, Callable, TYPE_CHECKING

from hls4ml.backends.xls.xls_types import XLSFunctionCall, XLSConst, XLSTensorVariable, XLSArrayType, XLSIntegerType, \
    XLSArray, XLSFixedPointType, float_to_significand, to_signed_fixed_precision
from hls4ml.model.types import FixedPrecisionType

if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer

from hls4ml.model.optimizer import OptimizerPass

from functools import wraps
import numpy as np


class XLSAttrBuilder:
    """A helper class that sets XLS specific attributes for the layers of the original ModelGraph.
    In doing so, we simplify the process of creating new optimization passes 
    and constructing the writer class. 
    The new attributes must be accessed with .get_attr(...)

    New attributes:
        - xls_module_name (str):                  DSLX module name (e.g. layer_4) used for the layer
        - xls_input_variable(XLSTensorVariable):  XLS representation of input shape and precision
        - xls_output_variable(XLSTensorVariable): XLS representation of output shape and precision
        - xls_weights(XLSArray):                  Weights converted to XLS array
        - xls_bias(XLSArray):                     Bias converted to XLS array
        - xls_func_call(XLSFunctionCall):         Function used for transformation, e.g. softmax_stable or conv2d

    Args:
        - node (Layer): A layer of the model graph
    """

    def __init__(self, node) -> None:
        self.node = node

    @staticmethod
    def attach_to_node(attr_name: Optional[str] = None):
        """A decorator-factory to easily chain 'set_attr' commands to the node.
        It calls the provided function. This eliminates a lot of boiler plate code.
        All the added attributes can be chained in one call since the wrapped function returns self.
        """

        def decorator(fn) -> Callable:
            name = attr_name or fn.__name__

            @wraps(fn)
            def wrapped(self, *args, **kwargs):
                val = fn(self, *args, **kwargs)
                assert name not in self.node.attributes, f"Duplicate attribute: '{name}'"
                self.node.set_attr(name, val)
                return self

            return wrapped

        return decorator

    @attach_to_node()
    def xls_weights(self) -> XLSConst | None:
        weights = self.node.weights.get('weight', None)
        if not weights:
            return None

        input_var = self.node.get_input_variable()
        output_var = self.node.get_output_variable()

        in_dim: int = input_var.shape[0]
        out_dim: int = output_var.shape[0]

        precision: FixedPrecisionType = to_signed_fixed_precision(weights.type.precision)

        match self.node.class_name:
            case 'Conv2D':
                n_chan = self.node.get_attr('n_chan')
                filt_height = self.node.get_attr('filt_height')
                filt_width = self.node.get_attr('filt_width')
                n_filt = self.node.get_attr('n_filt')
                mat = np.array(weights.data).reshape(filt_height, filt_width, n_chan, n_filt)
                mat_T = np.transpose(mat, (3, 2, 0, 1))  # in Keras the weights are transposed
            case 'Dense':
                mat = np.array(weights.data).reshape(in_dim, out_dim)
                mat_T = mat.T  # in Keras the weights are transposed
            case _:
                raise ValueError(f'Unsupported weights for layer {self.node.class_name}')
        shape = mat_T.shape
        xls_raw_array = XLSArray(
            array_type=XLSArrayType(
                element_type=XLSIntegerType(precision.width, signed=True),
                shape=mat_T.shape),
            array=float_to_significand(mat_T, precision))
        xls_precision = XLSFixedPointType.from_precision(precision)
        xls_fixed_point_array = XLSFunctionCall(
            name=f'fixed_point_util::make_fixed_points_{len(shape)}d',
            params=[xls_precision.binary_exponent],
            args=[xls_raw_array]
        )
        return XLSConst(
            name='WEIGHTS',
            value=xls_fixed_point_array,
            type=XLSArrayType(element_type=xls_precision, shape=shape)
        )

    @attach_to_node()
    def xls_bias(self) -> XLSConst | None:
        bias = self.node.weights.get('bias', None)
        if not bias:
            return None

        precision: FixedPrecisionType = to_signed_fixed_precision(bias.type.precision)
        raw_bias = float_to_significand(bias.data, precision)
        shape = raw_bias.shape
        xls_raw_array = XLSArray(
            array_type=XLSArrayType(element_type=XLSIntegerType(precision.width, signed=True), shape=shape),
            array=raw_bias
        )
        xls_precision = XLSFixedPointType.from_precision(precision)
        xls_fixed_point_array = XLSFunctionCall(
            name=f'fixed_point_util::make_fixed_points_{len(shape)}d',
            params=[xls_precision.binary_exponent],
            args=[xls_raw_array]
        )
        return XLSConst(
            name='BIAS',
            value=xls_fixed_point_array,
            type=XLSArrayType(element_type=xls_precision, shape=shape)
        )

    @attach_to_node()
    def xls_module_name(self) -> str:
        return f'layer_{self.node.index}'

    @attach_to_node()
    def xls_output_variable(self) -> XLSTensorVariable:
        return XLSTensorVariable.from_tensor_variable(name='OUTPUT', var=self.node.get_output_variable())

    @attach_to_node()
    def xls_input_variable(self, prev_layer: Layer | None) -> XLSTensorVariable:
        if not prev_layer:
            assert self.node.class_name == 'Input', \
                f'Unexpected class name for Layer {self.node.name}: {self.node.class_name}'
            assert self.node.get_input_variable() is None, \
                f'Input layer {self.node.name} should not have input variable'
            # Input and output are the same
            return XLSTensorVariable.from_tensor_variable(
                name='INPUT',
                var=self.node.get_output_variable()
            )

        # Import values from the previous layer, e.g.:
        # pub const INPUT_NUM_BITS = layer_1::OUTPUT_NUM_BUTS;
        prev_name = prev_layer.get_attr('xls_module_name')
        prev_var: XLSTensorVariable = prev_layer.get_attr('xls_output_variable')

        def qualified_name(xls_const: XLSConst):
            return f'{prev_name}::{xls_const.name}'

        return XLSTensorVariable(
            name='INPUT',
            num_bits=qualified_name(prev_var.num_bits),
            binary_exponent=qualified_name(prev_var.binary_exponent),
            shape=tuple(map(qualified_name, prev_var.shape)),
        )

    @attach_to_node()
    def xls_min_input_rank(self) -> int:
        """Minimally required rank of the input tensor.
         Input tensor can have a higher rank if it consists of multiple batches."""
        match self.node.class_name:
            case 'Conv2D':
                return 3
            case _:
                return 1

    @attach_to_node()
    def xls_func_call(self) -> XLSFunctionCall | str:
        name = None
        out_var = self.node.get_attr('xls_output_variable')
        params = []
        args = ['x']

        params_out = [out_var.num_bits.name, out_var.binary_exponent.name]
        params_rounding = ['ROUNDING_MODE', 'OVERFLOW_MODE']

        match self.node.class_name:

            # Input layer -> identity transformation
            case 'Input':
                return 'x'

            case 'Dense':
                name = f'dense::dense'
                args += ['WEIGHTS', 'BIAS']
                params = params_out + params_rounding

            case 'Conv2D':
                name = f'conv2d::conv2d_latency'
                args += ['WEIGHTS', 'BIAS']
                params = params_out + params_rounding

            case 'Activation':
                func_name = self.node.get_attr('activation').lower()
                name = f'activations::{func_name}'
                match func_name:
                    case 'relu':
                        params = []
                    case _:
                        raise ValueError(f'Unknown activation function {func_name}')

            case 'ParametrizedActivation':
                func_name = self.node.get_attr('activation').lower()
                name = f'activations::{func_name}'
                # TODO add extra parameters/arguments

            case 'Softmax':
                implementation = self.node.attributes.get('implementation', 'stable')
                params = params_out
                if implementation == 'stable':
                    name = f'activations::softmax_{implementation}'
                    params += params_rounding
                    args += ['EXP_NEG_TABLE', 'INV_TABLE']
                elif implementation == 'latency':
                    name = f'activations::softmax_{implementation}'
                    params += params_rounding
                    args += ['EXP_TABLE', 'INV_TABLE']
                elif implementation == 'argmax':
                    name = 'activations::argmax'
                # TODO: support implementation == 'legacy'
                else:
                    raise ValueError(f'Unknown softmax implementation {implementation}')
            case _:
                raise ValueError(f'Unknown layer type: {self.node.class_name}')
        return XLSFunctionCall(name=name, params=params, args=args)


class BuildAttr(OptimizerPass):
    """Builds the XLS-specific attributes for all layers.
    """

    def match(self, node: Layer) -> bool:
        if node.class_name == 'Input':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:
        prev_layer = None
        for layer in model.get_layers():
            try:
                # uses the builder to add all the attributes
                (XLSAttrBuilder(layer)
                 .xls_module_name()
                 .xls_min_input_rank()
                 .xls_input_variable(prev_layer)
                 .xls_output_variable()
                 .xls_weights()
                 .xls_bias()
                 .xls_func_call()
                 )
            except Exception as e:
                raise ValueError(f'Failed to build XLS attributes for layer {layer.name}: {e}')

            prev_layer = layer

        return False
