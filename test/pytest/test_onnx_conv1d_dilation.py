import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import hls4ml
from hls4ml.converters.onnx.convolution import parse_conv_layer


def make_onnx_conv1d(dilation, pads, stride=1, defaults=False, group=1):
    channels, filters, width, kernel = 2, 2 if group == 2 else 3, 17, 3
    effective = (kernel - 1) * (dilation or 1) + 1
    out_width = (width + sum(pads) - effective) // stride + 1
    weights = np.arange(filters * (channels // group) * kernel, dtype=np.float32)
    weights = weights.reshape(filters, channels // group, kernel) / 32 - 0.25
    bias = np.arange(filters, dtype=np.float32) / 8 - 0.125
    attrs = {} if defaults else dict(kernel_shape=[kernel], strides=[stride], pads=pads, group=group)
    if defaults and group != 1:
        attrs['group'] = group
    if dilation is not None:
        attrs['dilations'] = [dilation]
    node = helper.make_node('Conv', ['x', 'w', 'b'], ['y'], name='conv', **attrs)
    graph = helper.make_graph(
        [node],
        'dilated_conv',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, channels, width])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, filters, out_width])],
        [numpy_helper.from_array(weights, 'w'), numpy_helper.from_array(bias, 'b')],
    )
    reference = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(reference)
    # hls4ml expects the QONNX channels-last convention, including weights (M, K, C).
    converted = copy.deepcopy(reference)
    converted.graph.node[0].domain = 'qonnx.custom_op.channels_last'
    converted.graph.input[0].CopyFrom(helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, width, channels]))
    converted.graph.output[0].CopyFrom(helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, out_width, filters]))
    weights = weights.transpose(0, 2, 1).copy()
    converted.graph.initializer[0].CopyFrom(numpy_helper.from_array(weights, 'w'))
    converted.graph.value_info.extend(
        [
            helper.make_tensor_value_info('w', TensorProto.FLOAT, list(weights.shape)),
            helper.make_tensor_value_info('b', TensorProto.FLOAT, [filters]),
        ]
    )
    return reference, converted


@pytest.mark.parametrize('group', [1, 2])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'dilation,pads,stride,defaults',
    [
        (None, [0, 0], 1, True),
        (1, [0, 0], 2, False),
        (2, [0, 0], 1, False),
        (2, [1, 3], 2, False),
        (4, [4, 4], 1, False),
    ],
)
def test_onnx_conv1d_dilation(tmp_path, backend, dilation, pads, stride, defaults, group):
    reference, model = make_onnx_conv1d(dilation, pads, stride, defaults, group)
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        backend=backend,
        io_type='io_parallel',
        output_dir=str(tmp_path),
        hls_config={'Model': {'Precision': 'fixed<32,12>', 'ReuseFactor': 1}},
    )
    conv = next(layer for layer in hls_model.get_layers() if layer.class_name in ('Conv1D', 'DepthwiseConv1D'))
    assert conv.get_attr('dilation') == (dilation or 1)
    assert 'dilation_width' not in conv.attributes
    expected_shape = tuple(dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim)[1:]
    assert tuple(hls_model.get_output_variables()[0].shape) == expected_shape
    x = np.arange(34, dtype=np.float32).reshape(1, 2, 17) / 32 - 0.5
    expected = ort.InferenceSession(reference.SerializeToString(), providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
    hls_model.compile()
    actual = hls_model.predict(np.ascontiguousarray(x.transpose(0, 2, 1))).reshape(1, *expected_shape)
    np.testing.assert_allclose(actual, expected.transpose(0, 2, 1), atol=2e-3, rtol=0)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_onnx_conv1d_dilation_stream_rejected(tmp_path, backend):
    _, model = make_onnx_conv1d(2, [0, 0])
    with pytest.raises(NotImplementedError, match='only supported with io_parallel'):
        hls4ml.converters.convert_from_onnx_model(
            model,
            backend=backend,
            io_type='io_stream',
            output_dir=str(tmp_path),
            hls_config={'Model': {'Precision': 'fixed<32,12>', 'ReuseFactor': 1}},
        )


def test_onnx_conv2d_default_dilation():
    node = helper.make_node('Conv', ['x', 'w'], ['y'], name='conv', domain='qonnx.custom_op.channels_last')
    layer = parse_conv_layer(node, ['x', 'w'], [[1, 9, 11, 2], [3, 3, 3, 2]], None)
    assert layer['dilation_height'] == layer['dilation_width'] == 1
    assert (layer['out_height'], layer['out_width']) == (7, 9)
    assert 'dilation' not in layer
