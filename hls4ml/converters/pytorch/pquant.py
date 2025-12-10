from collections.abc import Iterable
from warnings import warn

import numpy as np

from hls4ml.converters.pytorch.convolution import parse_conv1d_layer, parse_conv2d_layer
from hls4ml.converters.pytorch.core import parse_batchnorm_layer, parse_linear_layer
from hls4ml.converters.pytorch.pooling import parse_pooling_layer
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.model.types import FixedPrecisionType


def extract_fixed_quantizer_config(q, shape, input, name):
    q_params = q._parameters

    shape = tuple(shape[1:])  # type: ignore
    print(f'FixedPointQuantizer shape: {shape}')
    if any([s is None for s in shape]):
        raise ValueError(f'Tensor {input} has at least one dimension with no fixed size')
    k, i, f = q_params['k'].data, q_params['i'].data, q_params['f'].data
    k, B, I = k, k + i + f, k + i  # type: ignore # noqa: E741
    k, B, I = k.detach().cpu().numpy(), B.detach().cpu().numpy(), I.detach().cpu().numpy()  # noqa: E741
    I = np.where(B > 0, I, 0)  # noqa: E741 # type: ignore

    k = np.broadcast_to(k.astype(np.int16), (1,) + shape)  # type: ignore
    B = np.broadcast_to(B.astype(np.int16), (1,) + shape)  # type: ignore
    I = np.broadcast_to(I.astype(np.int16), (1,) + shape)  # noqa: E741

    overflow_mode: str = q.overflow
    round_mode: str = q.round_mode
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]
    fusible = np.unique(k).size == 1 and np.unique(B).size == 1 and np.unique(I).size == 1

    return {
        'name': name,
        'inputs': [input],
        'class_name': 'FixedPointQuantizer',
        'mask_kbi': (k, B, I),
        'SAT': overflow_mode,
        'RND': round_mode,
        'fusible': fusible,
        'overrides': {},
    }


def add_quantizer_info(class_object, input_names, input_shapes, output_shape, layer):
    if getattr(class_object, 'quantize_input', False) and hasattr(class_object, 'input_quantizer'):
        if isinstance(class_object.input_quantizer, Iterable):
            iq_confs = [
                extract_fixed_quantizer_config(q, shape, input, f'{layer["name"]}_iq_{i}')
                for q, shape, input, i in zip(
                    class_object.input_quantizer, input_shapes, input_names, [k for k in range(len(input_names))]
                )
            ]
        else:
            iq_confs = [
                extract_fixed_quantizer_config(
                    class_object.input_quantizer, input_shapes[0], input_names[0], f'{layer["name"]}_iq'
                )
            ]
        layer['inputs'] = [q['name'] for q in iq_confs]
        iq_shapes = input_shapes
    else:
        iq_confs = []
        iq_shapes = []

    if getattr(class_object, 'quantize_output', False) and hasattr(class_object, 'output_quantizer'):
        if isinstance(class_object.output_quantizer, Iterable):
            oq_confs = [
                extract_fixed_quantizer_config(q, output_shape, layer['name'], f'{layer["name"]}_oq_{i}')
                for q, i in zip(class_object.output_quantizer, [k for k in range(len(class_object.output_quantizer))])
            ]
            oq_shapes = [output_shape for _ in len(class_object.output_quantizer)]
        else:
            oq_confs = [
                extract_fixed_quantizer_config(
                    class_object.output_quantizer, output_shape, layer['name'], f'{layer["name"]}_oq'
                )
            ]
            oq_shapes = [output_shape]
    else:
        oq_confs = []
        oq_shapes = []

    out_shapes = []
    if iq_shapes:
        out_shapes.append(iq_shapes)
    out_shapes.append(output_shape)
    if oq_shapes:
        out_shapes.append(oq_shapes)

    return iq_confs + [layer] + oq_confs, iq_shapes + [output_shape] + oq_shapes


def make_pquant_handler(base_parse_func, op, op_check=None):
    if op_check is None:
        op_check = op

    @pytorch_handler(op)
    def handler(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
        assert op in operation
        layer, output_shape = base_parse_func(
            op_check, layer_name, input_names, input_shapes, node, class_object, data_reader, config
        )
        layers, output_shapes = add_quantizer_info(class_object, input_names, input_shapes, output_shape, layer)
        return layers, output_shapes

    handler.__name__ = f'parse_{op.lower()}_layer'
    return handler


parse_pqlinear_layer = make_pquant_handler(parse_linear_layer, 'PQDense', 'PQLinear')
parse_pqbatchnorm_layer = make_pquant_handler(parse_batchnorm_layer, 'PQBatchNorm2d')
parse_pqconv1d_layer = make_pquant_handler(parse_conv1d_layer, 'PQConv1d')
parse_pqconv2d_layer = make_pquant_handler(parse_conv2d_layer, 'PQConv2d')
parse_pqpool1d_layer = make_pquant_handler(parse_pooling_layer, 'PQAvgPool1d', 'AvgPool1d')
parse_pqpool2d_layer = make_pquant_handler(parse_pooling_layer, 'PQAvgPool2d', 'AvgPool2d')


def parse_quant_activation_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    layer = {}

    layer['activation'] = class_object.activation_name

    print(f'Parsing activation: {layer["activation"]}')

    layer['name'] = layer_name
    layer['inputs'] = input_names

    if layer['activation'] == 'hard_tanh':
        layer['class_name'] = 'HardActivation'
        layer['slope'] = 0.5
        layer['shift'] = 0.5
        layer['slope_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        layer['shift_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        warn(f'Hard Tanh activation {layer_name} is currently not supported for bit-exactness.')

    elif layer['activation'] == 'relu' and class_object.use_multiplier:
        raise Exception('hls4ml does not currently support activations with multiplier')
        """
        layer['activation'] = 'multiplier_relu'
        layer['class_name'] = 'MultiplierReLU'
        layer['param_data'] = class_object.multiplier.data.numpy()
        """

    else:
        layer['class_name'] = 'Activation'

    output_shape = input_shapes[0]
    return layer, output_shape


parse_pqactivation_layer = make_pquant_handler(parse_quant_activation_layer, 'PQActivation')
