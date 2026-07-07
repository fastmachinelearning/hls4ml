# Typing imports
from __future__ import annotations  # makes all annotations into strings

from typing import TYPE_CHECKING

from hls4ml.backends.xls.xls_types import (
    XLSConstDefinition,
    XLSFixedPointDefinition,
    XLSFunctionCallDefinition,
    XLSQualifiedName,
    XLSTensorVariableDefinition,
)

if TYPE_CHECKING:
    from hls4ml.model.layers import Layer


def xls_weights_key(node: Layer) -> str:
    class_name = node.class_name
    if class_name == 'ApplyAlpha':
        class_name = 'BatchNormalization'
    match class_name:
        case 'PReLU':
            return 'param'
        case 'BatchNormalization':
            return 'scale'
        case _:
            return 'weight'


def xls_weights(node: Layer) -> XLSConstDefinition | None:
    return node.weights.get(xls_weights_key(node), None)


def xls_bias(node: Layer) -> XLSConstDefinition | None:
    return node.weights.get('bias', None)


def xls_module_name(node: Layer) -> str:
    name = ''.join(c for c in node.name if c.isalnum() or c == '_').lower()
    return f'layer_{node.index}_{name}'


def xls_input_variables(node: Layer) -> list[XLSTensorVariableDefinition]:
    if node.class_name == 'Input':
        assert node.get_input_variable() is None, f'Input layer {node.name} should not have input variable'
        return [(node.get_output_variable())]
    else:
        return [node.get_input_variable(name) for name in node.inputs]


def xls_min_input_rank(node: Layer) -> int:
    """Minimally required rank of the input tensor.
    Input tensor can have a higher rank if it consists of multiple batches.
    NB: in the case of multiple input variables, the rank is determined by the first input variable.
    """
    name = node.class_name
    if name.endswith('2D'):
        return 3
    elif name.endswith('1D'):
        return 2
    elif name in ('Reshape', 'Concatenate'):
        return len(node.get_input_variable().shape)
    elif name == 'Transpose':
        return len(node.get_attr('perm'))
    else:
        return 1


def xls_extra_func_params(node: Layer) -> list[XLSConstDefinition]:
    layer = node
    class_name = layer.class_name
    if class_name == 'Concatenate':
        rank = len(layer.get_input_variable().shape)
        if rank == 1:
            return []
        axis = layer.get_attr('axis')
        if axis > 0:
            # Convert axis to a 0-based index.
            # This is the same adjustment as in hls4ml.model.layers.Concatenate.initialize()
            # TODO: should it be done earlier, when converting from frontend?
            axis -= 1
        if axis == -1:
            axis = rank - 1
        return [XLSConstDefinition(name='AXIS', value=axis, type='u32')]
    elif class_name in ('Conv1D', 'DepthwiseConv1D'):
        return [
            XLSConstDefinition(name='STRIDE', value=layer.get_attr('stride_width'), type='u32'),
            XLSConstDefinition(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
            XLSConstDefinition(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
            XLSConstDefinition(
                name='DATA_FORMAT', value=f'data_format::DataFormat::{layer.get_attr("data_format").upper()}'
            ),
        ]
    elif class_name in ('Conv2D', 'DepthwiseConv2D'):
        return [
            XLSConstDefinition(name='STRIDE_HEIGHT', value=layer.get_attr('stride_height'), type='u32'),
            XLSConstDefinition(name='STRIDE_WIDTH', value=layer.get_attr('stride_width'), type='u32'),
            XLSConstDefinition(name='PAD_TOP', value=layer.get_attr('pad_top'), type='u32'),
            XLSConstDefinition(name='PAD_BOTTOM', value=layer.get_attr('pad_bottom'), type='u32'),
            XLSConstDefinition(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
            XLSConstDefinition(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
            XLSConstDefinition(
                name='DATA_FORMAT', value=f'data_format::DataFormat::{layer.get_attr("data_format").upper()}'
            ),
        ]
    elif 'Pooling' in class_name:
        pool_op = f'pooling::PoolingOperation::{layer.get_attr("pool_op").upper()}'
        data_format = f'data_format::DataFormat::{layer.get_attr("data_format").upper()}'
        if class_name.startswith('GlobalPooling'):
            return [
                XLSConstDefinition(name='POOL_OP', value=pool_op),
                XLSConstDefinition(name='DATA_FORMAT', value=data_format),
            ]
        elif class_name.endswith('Pooling1D'):
            count_pad = str(layer.get_attr('count_pad')).lower()
            return [
                XLSConstDefinition(name='POOL_OP', value=pool_op),
                XLSConstDefinition(name='POOL_SIZE', value=layer.get_attr('pool_width'), type='u32'),
                XLSConstDefinition(name='STRIDE', value=layer.get_attr('stride_width'), type='u32'),
                XLSConstDefinition(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                XLSConstDefinition(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                XLSConstDefinition(name='COUNT_PAD', value=count_pad, type='bool'),
                XLSConstDefinition(name='DATA_FORMAT', value=data_format),
            ]
        elif class_name.endswith('Pooling2D'):
            count_pad = str(layer.get_attr('count_pad')).lower()
            return [
                XLSConstDefinition(name='POOL_OP', value=pool_op),
                XLSConstDefinition(name='POOL_HEIGHT', value=layer.get_attr('pool_height'), type='u32'),
                XLSConstDefinition(name='POOL_WIDTH', value=layer.get_attr('pool_width'), type='u32'),
                XLSConstDefinition(name='STRIDE_HEIGHT', value=layer.get_attr('stride_height'), type='u32'),
                XLSConstDefinition(name='STRIDE_WIDTH', value=layer.get_attr('stride_width'), type='u32'),
                XLSConstDefinition(name='PAD_TOP', value=layer.get_attr('pad_top'), type='u32'),
                XLSConstDefinition(name='PAD_BOTTOM', value=layer.get_attr('pad_bottom'), type='u32'),
                XLSConstDefinition(name='PAD_LEFT', value=layer.get_attr('pad_left'), type='u32'),
                XLSConstDefinition(name='PAD_RIGHT', value=layer.get_attr('pad_right'), type='u32'),
                XLSConstDefinition(name='COUNT_PAD', value=count_pad, type='bool'),
                XLSConstDefinition(name='DATA_FORMAT', value=data_format),
            ]
        else:
            raise ValueError(f'Unsupported pooling layer {class_name}')
    elif class_name == 'Reshape':
        assert len(layer.outputs) == 1, f'Reshape layer should have exactly one output variable, got {layer.outputs}'
        return list(layer.get_output_variable().xls_dims)
    elif class_name == 'Transpose':
        return [
            XLSConstDefinition(name=f'PERM_{i}', value=perm, type='u32') for i, perm in enumerate(layer.get_attr('perm'))
        ]
    else:
        return []


def xls_extra_func_args(node: Layer) -> list[XLSConstDefinition]:
    layer = node
    match layer.class_name:
        case 'HardActivation':
            return [
                XLSConstDefinition(
                    name=arg_name.upper(),
                    value=XLSFixedPointDefinition.from_float(
                        layer.get_attr(arg_name), precision=layer.get_attr(f'{arg_name}_t').precision
                    ),
                )
                for arg_name in ['slope', 'shift']
            ]
        case 'ParametrizedActivation':
            precision = layer.get_attr('param_t').precision
            value = layer.get_attr('activ_param')
            if layer.get_attr('activation').lower() in ('leakyrelu', 'leaky_relu', 'thresholdedrelu'):
                return [
                    XLSConstDefinition(name='ACTIVATION_PARAM', value=XLSFixedPointDefinition.from_float(value, precision))
                ]
        case _:
            pass
    return []


def xls_func_name(node: Layer) -> XLSQualifiedName:
    match node.class_name:
        case 'Input':
            # Identity transformation except for OverflowMode::SAT_SYM case.
            return XLSQualifiedName(name='resize_1d', module_name='fixed_point_util')
        case 'ApplyAlpha':
            return XLSQualifiedName(name='normalize', module_name='batchnorm')
        case 'BatchNormalization':
            return XLSQualifiedName(name='normalize', module_name='batchnorm')
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
            op = node.get_attr('op').lower()
            return XLSQualifiedName(name=op, module_name='merge')
        case 'Concatenate':
            rank = len(node.get_input_variable().shape)
            return XLSQualifiedName(name=f'concatenate{rank}d', module_name='merge')
        case 'Dot':
            return XLSQualifiedName(name='dot', module_name='merge')
        case 'Activation':
            return XLSQualifiedName(name=node.get_attr('activation').lower(), module_name='activations')
        case 'HardActivation':
            return XLSQualifiedName(name=node.get_attr('activation').lower(), module_name='activations')
        case 'ParametrizedActivation':
            return XLSQualifiedName(name=node._get_act_function_name(), module_name='activations')
        case 'PReLU':
            return XLSQualifiedName(name='prelu', module_name='activations')
        case 'Reshape':
            in_shape = node.get_input_variable().shape
            out_shape = node.get_output_variable().shape
            name = f'reshape_{len(in_shape)}d_to_{len(out_shape)}d'
            return XLSQualifiedName(name=name, module_name='reshape')
        case 'Softmax':
            implementation = node.attributes.get('implementation', 'stable')
            match implementation:
                case 'stable':
                    name = 'softmax_stable'
                case 'latency':
                    name = 'softmax_latency'
                case 'argmax':
                    name = 'argmax'
                case _:
                    # TODO: support implementation == 'legacy'
                    raise ValueError(f'Unknown softmax implementation {implementation}')
            return XLSQualifiedName(name=name, module_name='activations')
        case 'Transpose':
            rank = len(node.get_input_variable().shape)
            return XLSQualifiedName(name=f'transpose_{rank}d', module_name='transpose')
        case 'TernaryTanh':
            return XLSQualifiedName(name='ternary_tanh', module_name='activations')
        case _:
            raise ValueError(f'Unknown layer type: {node.class_name}')


def xls_func_call(node: Layer) -> XLSFunctionCallDefinition:
    in_vars = xls_input_variables(node)
    out_vars = node.get_variables()
    name = xls_func_name(node)
    params = [
        x.xls_name
        for out_var in out_vars
        for x in (
            out_var.xls_num_bits,
            out_var.xls_binary_exponent,
            out_var.xls_rounding_mode,
            out_var.xls_overflow_mode,
        )
    ] + [x.xls_name for x in xls_extra_func_params(node)]
    args = [f'x_{i}' for i in range(len(in_vars))]
    args += [w.xls_name for w in [xls_weights(node), xls_bias(node)] if w is not None]
    args += [x.lookup_table.xls_name for x in node.get_attr('lookup_tables', [])]
    args += [x.xls_name for x in xls_extra_func_args(node)]
    return XLSFunctionCallDefinition(name=name, params=params, args=args)
