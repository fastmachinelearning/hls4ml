from pathlib import Path

import numpy as np
import onnx
import pytest
import qonnx.core.onnx_exec as oxe
import torch
import torch.nn as nn
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.util.cleanup import cleanup_model

import hls4ml

test_root_path = Path(__file__).parent


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(16, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 32)
        self.linear5 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x3 = self.linear3(x)
        x = self.linear4(x3)
        x = self.linear5(x)

        return x, x3


@pytest.fixture
def onnx_model(tmp_path):
    model = LinearModel()

    x = torch.from_numpy(np.random.rand(1, 16))
    model(x.float())

    onnx_file = tmp_path / 'test_multiout.onnx'

    torch.onnx.export(model, x.float(), onnx_file)

    onnx_model = ModelWrapper(onnx.load(onnx_file))
    onnx_model = cleanup_model(onnx_model)
    onnx_model = onnx_model.transform(GemmToMatMul())
    onnx_model = cleanup_model(onnx_model)

    return onnx_model


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_multiout_onnx(onnx_model, io_type):

    X = np.random.rand(1, 16)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)

    idict = {onnx_model.graph.input[0].name: X}
    exe_out = oxe.execute_onnx(onnx_model, idict)
    y_qonnx = [exe_out[onnx_model.graph.output[1].name], exe_out[onnx_model.graph.output[0].name]]

    config = hls4ml.utils.config_from_onnx_model(
        onnx_model, granularity='name', default_precision='fixed<32, 16>', backend='Vitis'
    )

    output_dir = str(test_root_path / f'hls4mlprj_multiout_onnx_{io_type}')

    hls_model = hls4ml.converters.convert_from_onnx_model(
        onnx_model,
        hls_config=config,
        io_type=io_type,
        output_dir=output_dir,
        backend='Vitis',
    )

    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))

    np.testing.assert_allclose(y_qonnx[0].ravel(), y_hls4ml[0].ravel(), atol=1e-4, rtol=0)
    np.testing.assert_allclose(y_qonnx[1].ravel(), y_hls4ml[1].ravel(), atol=1e-4, rtol=0)
