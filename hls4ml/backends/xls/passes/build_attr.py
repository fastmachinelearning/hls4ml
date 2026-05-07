# Typing imports
from __future__ import annotations  # makes all annotations into strings

from typing import Literal, Optional, Callable, TYPE_CHECKING

from hls4ml.backends.xls.xls_types import XLSFunctionCall, XLSConst, XLSTensorVariable, XLSArrayType, XLSIntegerType, \
    XLSArray, XLSFixedPointType, float_to_significand, to_signed_fixed_precision, XLSFixedPoint, XLSQualifiedName
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
        - xls_module_name (str):                  DSLX module name (e.g. layer_4_softmax) used for the layer
        - xls_input_variables(list[XLSTensorVariable]):  XLS representation of input shape and precision
        - xls_output_variables(list[XLSTensorVariable]): XLS representation of output shape and precision
        - xls_weights(XLSArray):                  Weights converted to XLS array
        - xls_bias(XLSArray):                     Bias converted to XLS array
        - xls_extra_func_params(list[XLSConst]):  Extra parameters for function call, e.g. stride, padding, pool_op, etc.
        - xls_extra_func_args(list[XLSConst]):    Extra arguments for function call, e.g. activation parameter.
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

    @staticmethod
    def _xls_const_array(name: str, data: np.ndarray, precision: FixedPrecisionType) -> XLSConst:
        xls_raw_array = XLSArray(
            array_type=XLSArrayType(
                element_type=XLSIntegerType(precision.width, signed=True),
                shape=data.shape),
            array=float_to_significand(data, precision))
        xls_precision = XLSFixedPointType.from_precision(precision)
        xls_fixed_point_array = XLSFunctionCall(
            name=f'fixed_point_util::make_fixed_points_{len(data.shape)}d',
            params=[xls_precision.binary_exponent],
            args=[xls_raw_array]
        )
        return XLSConst(
            name=name,
            value=xls_fixed_point_array,
            type=XLSArrayType(element_type=xls_precision, shape=data.shape)
        )

    @attach_to_node()
    def xls_weights(self) -> XLSConst | None:
        precision = None
        if self.node.class_name == 'PReLU':
            weights = self.node.weights.get('param_data')
            precision = self.node.get_attr('param_t').precision
        else:
            weights = self.node.weights.get('weight', None)
        if weights is None:
            return None

        precision: FixedPrecisionType = to_signed_fixed_precision(precision or weights.type.precision)

        input_var = self.node.get_input_variable()
        output_var = self.node.get_output_variable()

        match self.node.class_name:
            case 'Conv1D':
                data = np.asarray(weights.data)
                expected_shape = tuple(self.node.get_attr(x) for x in ['filt_width', 'n_chan', 'n_filt'])
            case 'DepthwiseConv1D':
                data = np.asarray(weights.data)
                expected_shape = tuple(self.node.get_attr(x) for x in ['filt_width', 'n_chan', 'depth_multiplier'])
            case 'Conv2D':
                data = np.asarray(weights.data)
                expected_shape = tuple(self.node.get_attr(x) for x in ['filt_height', 'filt_width', 'n_chan', 'n_filt'])
            case 'DepthwiseConv2D':
                data = np.asarray(weights.data)
                expected_shape = tuple(
                    self.node.get_attr(x) for x in ['filt_height', 'filt_width', 'n_chan', 'depth_multiplier']
                )
            case 'Dense':
                # Transpose the weights so that we can call dot_prod(x, w[i]) in dense.x
                data = np.asarray(weights.data).T
                expected_shape = (output_var.shape[0], input_var.shape[0])
            case 'PReLU':
                data = weights
                expected_shape = (input_var.shape[0],)
            case _:
                raise ValueError(f'Unsupported weights for layer {self.node.class_name}')

        assert data.shape == expected_shape, \
            f'Weights shape mismatch: expected {expected_shape}, got {data.shape}'

        return XLSAttrBuilder._xls_const_array(name='WEIGHTS', data=data, precision=precision)

    @attach_to_node()
    def xls_bias(self) -> XLSConst | None:
        bias = self.node.weights.get('bias', None)
        if not bias:
            return None

        precision: FixedPrecisionType = to_signed_fixed_precision(bias.type.precision)
        return XLSAttrBuilder._xls_const_array(name='BIAS', data=bias.data, precision=precision)

    @attach_to_node()
    def xls_module_name(self) -> str:
        name = ''.join(c for c in self.node.name if c.isalnum() or c == '_').lower()
        return f'layer_{self.node.index}_{name}'

    @attach_to_node()
    def xls_output_variables(self) -> list[XLSTensorVariable]:
        return [
            XLSTensorVariable.from_tensor_variable(self.node.get_output_variable(name))
            for (i, name) in enumerate(self.node.outputs)
        ]

    @attach_to_node()
    def xls_input_variables(self) -> list[XLSTensorVariable]:
        if self.node.class_name == 'Input':
            assert self.node.get_input_variable() is None, \
                f'Input layer {self.node.name} should not have input variable'
            out_var = self.node.get_output_variable()
            return [XLSTensorVariable.from_tensor_variable(out_var, name=f'input_{out_var.name}')]
        else:
            return [
                XLSTensorVariable.from_tensor_variable(var=self.node.get_input_variable(name))
                for name in self.node.inputs
            ]

    @attach_to_node()
    def xls_min_input_rank(self) -> int:
        """Minimally required rank of the input tensor.
         Input tensor can have a higher rank if it consists of multiple batches.
         NB: in the case of multiple input variables, the rank is determined by the first input variable.
         """
        name = self.node.class_name
        if name.endswith('2D'):
            return 3
        elif name.endswith('1D'):
            return 2
        elif name == 'Reshape':
            return len(self.node.get_input_variable().shape)
        elif name == 'Transpose':
            return len(self.node.get_attr('perm'))
        else:
            return 1

    @attach_to_node()
    def xls_extra_func_params(self) -> list[XLSConst]:
        layer = self.node
        class_name = layer.class_name
        if class_name in ('Conv1D', 'DepthwiseConv1D'):
            return [
                XLSConst(name='STRIDE', value=layer.get_attr('stride_width'), type='u32'),
                XLSConst(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                XLSConst(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                XLSConst(name='DATA_FORMAT',
                         value=f"data_format::DataFormat::{layer.get_attr('data_format').upper()}")
            ]
        elif class_name in ('Conv2D', 'DepthwiseConv2D'):
            return [
                XLSConst(name='STRIDE_HEIGHT', value=layer.get_attr('stride_height'), type='u32'),
                XLSConst(name='STRIDE_WIDTH', value=layer.get_attr('stride_width'), type='u32'),
                XLSConst(name='PAD_TOP', value=layer.get_attr('pad_top'), type='u32'),
                XLSConst(name='PAD_BOTTOM', value=layer.get_attr('pad_bottom'), type='u32'),
                XLSConst(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                XLSConst(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                XLSConst(name='DATA_FORMAT',
                         value=f"data_format::DataFormat::{layer.get_attr('data_format').upper()}")
            ]
        elif 'Pooling' in class_name:
            pool_op = f"pooling::PoolingOperation::{layer.get_attr('pool_op').upper()}"
            data_format = f"data_format::DataFormat::{layer.get_attr('data_format').upper()}"
            if class_name.startswith('GlobalPooling'):
                return [
                    XLSConst(name='POOL_OP', value=pool_op),
                    XLSConst(name='DATA_FORMAT', value=data_format)
                ]
            elif class_name.endswith('Pooling1D'):
                count_pad = str(layer.get_attr('count_pad')).lower()
                return [
                    XLSConst(name='POOL_OP', value=pool_op),
                    XLSConst(name='POOL_SIZE', value=layer.get_attr('pool_width'), type='u32'),
                    XLSConst(name='STRIDE', value=layer.get_attr('stride_width'), type='u32'),
                    XLSConst(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                    XLSConst(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                    XLSConst(name='COUNT_PAD', value=count_pad, type='bool'),
                    XLSConst(name='DATA_FORMAT', value=data_format)
                ]
            elif class_name.endswith('Pooling2D'):
                count_pad = str(layer.get_attr('count_pad')).lower()
                return [
                    XLSConst(name='POOL_OP', value=pool_op),
                    XLSConst(name='POOL_HEIGHT', value=layer.get_attr('pool_height'), type='u32'),
                    XLSConst(name='POOL_WIDTH', value=layer.get_attr('pool_width'), type='u32'),
                    XLSConst(name='STRIDE_HEIGHT', value=layer.get_attr('stride_height'), type='u32'),
                    XLSConst(name='STRIDE_WIDTH', value=layer.get_attr('stride_width'), type='u32'),
                    XLSConst(name='PAD_TOP', value=layer.get_attr('pad_top'), type='u32'),
                    XLSConst(name='PAD_BOTTOM', value=layer.get_attr('pad_bottom'), type='u32'),
                    XLSConst(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                    XLSConst(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                    XLSConst(name='COUNT_PAD', value=count_pad, type='bool'),
                    XLSConst(name='DATA_FORMAT', value=data_format)
                ]
            else:
                raise ValueError(f'Unsupported pooling layer {class_name}')
        elif class_name == 'Reshape':
            out_vars = layer.get_attr('xls_output_variables')
            assert len(out_vars) == 1, f"Reshape layer should have exactly one output variable, got {len(out_vars)}"
            return list(out_vars[0].shape)
        elif class_name == 'Transpose':
            return [
                XLSConst(name=f'PERM_{i}', value=perm, type=f'u32')
                for i, perm in enumerate(layer.get_attr('perm'))
            ]
        else:
            return []

    @attach_to_node()
    def xls_extra_func_args(self) -> list[XLSConst]:
        layer = self.node
        match layer.class_name:
            case 'HardActivation':
                return [
                    XLSConst(
                        name=arg_name.upper(),
                        value=XLSFixedPoint.from_float(
                            layer.get_attr(arg_name),
                            precision=to_signed_fixed_precision(layer.get_attr(f'{arg_name}_t').precision))
                    )
                    for arg_name in ['slope', 'shift']
                ]
            case 'ParametrizedActivation':
                precision = to_signed_fixed_precision(layer.get_attr('param_t').precision)
                value = layer.get_attr('activ_param')
                if layer.get_attr('activation').lower() in ('leakyrelu', 'leaky_relu', 'thresholdedrelu'):
                    return [XLSConst(
                        name='ACTIVATION_PARAM',
                        value=XLSFixedPoint.from_float(value, precision))]
            case _:
                pass
        return []

    @staticmethod
    def func_name(layer: Layer) -> XLSQualifiedName:
        class_name = layer.class_name
        match class_name:
            case 'Input':
                # Identity transformation except for OverflowMode::SAT_SYM case.
                return XLSQualifiedName(name='resize_1d', module_name='fixed_point_util')
            case 'Dense':
                return XLSQualifiedName(name='dense', module_name='dense')
            case 'Conv1D':
                return XLSQualifiedName(name='conv1d_latency', module_name='conv1d')
            case 'DepthwiseConv1D':
                return XLSQualifiedName(name='depthwise_conv_1d', module_name='depthwise_conv')
            case 'Conv2D':
                return XLSQualifiedName(name='conv2d_latency', module_name='conv2d')
            case 'DepthwiseConv2D':
                return XLSQualifiedName(name='depthwise_conv_2d', module_name='depthwise_conv')
            case 'Pooling1D':
                return XLSQualifiedName(name='pooling_1d', module_name='pooling')
            case 'Pooling2D':
                return XLSQualifiedName(name='pooling_2d', module_name='pooling')
            case 'GlobalPooling1D':
                return XLSQualifiedName(name='global_pooling_1d', module_name='pooling')
            case 'GlobalPooling2D':
                return XLSQualifiedName(name='global_pooling_2d', module_name='pooling')
            case 'Merge':
                op = layer.get_attr('op').lower()
                return XLSQualifiedName(name=op, module_name='merge')
            case 'Dot':
                return XLSQualifiedName(name='dot', module_name='merge')
            case 'Activation':
                return XLSQualifiedName(name=layer.get_attr('activation').lower(), module_name='activations')
            case 'HardActivation':
                return XLSQualifiedName(name=layer.get_attr('activation').lower(), module_name='activations')
            case 'ParametrizedActivation':
                return XLSQualifiedName(name=layer._get_act_function_name(), module_name='activations')
            case 'PReLU':
                return XLSQualifiedName(name='prelu', module_name='activations')
            case 'Reshape':
                in_shape = layer.get_input_variable().shape
                out_shape = layer.get_output_variable().shape
                name = f'reshape_{len(in_shape)}d_to_{len(out_shape)}d'
                return XLSQualifiedName(name=name, module_name='reshape')
            case 'Softmax':
                implementation = layer.attributes.get('implementation', 'stable')
                match implementation:
                    case 'stable':
                        name = f'softmax_stable'
                    case 'latency':
                        name = f'softmax_latency'
                    case 'argmax':
                        name = 'argmax'
                    case _:
                        # TODO: support implementation == 'legacy'
                        raise ValueError(f'Unknown softmax implementation {implementation}')
                return XLSQualifiedName(name=name, module_name='activations')
            case 'Transpose':
                rank = len(layer.get_input_variable().shape)
                return XLSQualifiedName(name=f'transpose_{rank}d', module_name='transpose')
            case 'TernaryTanh':
                return XLSQualifiedName(name='ternary_tanh', module_name='activations')
            case _:
                raise ValueError(f'Unknown layer type: {layer.class_name}')

    @attach_to_node()
    def xls_func_call(self) -> XLSFunctionCall:
        in_vars = self.node.get_attr('xls_input_variables')
        out_vars = self.node.get_attr('xls_output_variables')
        name = self.func_name(self.node)
        params = [x.name for out_var in out_vars for x in (
            out_var.num_bits,
            out_var.binary_exponent,
            out_var.rounding_mode,
            out_var.overflow_mode,
        )] + [x.name for x in self.node.get_attr('xls_extra_func_params')]
        args = [f'x_{i}' for i in range(len(in_vars))]
        args += [self.node.get_attr(x).name
                 for x in ('xls_weights', 'xls_bias')
                 if self.node.get_attr(x) is not None]
        args += [x.lookup_table.name for x in self.node.get_attr('lookup_tables', [])]
        args += [x.name for x in self.node.get_attr('xls_extra_func_args')]
        return XLSFunctionCall(name=name, params=params, args=args)


class BuildAttr(OptimizerPass):
    """Builds the XLS-specific attributes for all layers.
    """

    def match(self, node: Layer) -> bool:
        return True

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:
        try:
            # uses the builder to add all the attributes
            (XLSAttrBuilder(node)
             .xls_module_name()
             .xls_min_input_rank()
             .xls_input_variables()
             .xls_output_variables()
             .xls_weights()
             .xls_bias()
             .xls_extra_func_params()
             .xls_extra_func_args()
             .xls_func_call()
             )
        except Exception as e:
            raise ValueError(
                f'Failed to build XLS attributes for layer (name={node.name}, class_name={node.class_name}): {e}')
        return False
