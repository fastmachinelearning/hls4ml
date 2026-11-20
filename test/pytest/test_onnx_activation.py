from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from hls4ml.converters import convert_from_onnx_model
from hls4ml.utils.config import config_from_onnx_model

test_root_path = Path(__file__).parent


def _make_activation_onnx(op_type, **attrs):
    input_tensor = helper.make_tensor_value_info('global_in', TensorProto.FLOAT, [1, 8])
    output_tensor = helper.make_tensor_value_info('global_out', TensorProto.FLOAT, [1, 8])
    node = helper.make_node(op_type, name='act', inputs=['global_in'], outputs=['global_out'], **attrs)
    graph = helper.make_graph([node], 'ActivationGraph', [input_tensor], [output_tensor])
    return helper.make_model(graph)


def _elu(x, alpha):
    return np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0)))


def _leaky_relu(x, alpha):
    return np.where(x >= 0, x, alpha * x)


def _thresholded_relu(x, alpha):
    return np.where(x > alpha, x, 0.0)


@pytest.mark.parametrize(
    'op_type,attrs,reference,alpha',
    [
        ('Elu', {}, _elu, 1.0),
        ('Elu', {'alpha': 0.5}, _elu, 0.5),
        ('LeakyRelu', {}, _leaky_relu, 0.01),
        ('LeakyRelu', {'alpha': 0.25}, _leaky_relu, 0.25),
        ('ThresholdedRelu', {}, _thresholded_relu, 1.0),
        ('ThresholdedRelu', {'alpha': 0.5}, _thresholded_relu, 0.5),
    ],
)
def test_onnx_activation_alpha(test_case_id, op_type, attrs, reference, alpha):
    model = _make_activation_onnx(op_type, **attrs)
    config = config_from_onnx_model(model, granularity='name', default_precision='fixed<32,16>')
    hls_model = convert_from_onnx_model(model, output_dir=str(test_root_path / test_case_id), hls_config=config)
    hls_model.compile()

    data = np.arange(-3.0, 3.0, 0.25).reshape(-1, 8).astype(np.float32) + 0.125
    pred = hls_model.predict(data)

    np.testing.assert_allclose(pred.reshape(data.shape), reference(data, alpha), rtol=0, atol=0.02)
