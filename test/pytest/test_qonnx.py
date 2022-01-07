#!/usr/bin/env python
import pytest
import hls4ml
import numpy as np
import qonnx.util.cleanup
import urllib
import os
# To conveniently run QONNX inference
from finn.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe

def test_tfc_2w2a():
    # download test model
    dl_dir = "./"
    dl_file = dl_dir + "qonnx-tfc-2w2a.onnx"
    tfc_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_2W2A.onnx"
    )
    urllib.request.urlretrieve(tfc_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = dl_dir + "/qonnx-tfc-2w2a-clean.onnx"

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)

    # Execute QONNX model inference
    # TODO make the test bigger
    ishape = (1,1,28,28)
    np.random.seed(0)
    X = np.random.uniform(low=-1, high=+1, size=np.product(ishape)).reshape(ishape).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model)
    # Some hand-derived config
    # TODO should be auto-derived by QuantizeDenseOutput pass after some adaptation
    config['LayerName'] = {}
    config['LayerName']['global_in'] = {'Precision' : 'ap_fixed<16,2>'}
    config['LayerName']['Dense_MatMul_0'] = {'Precision' : {'accum' : 'ap_int<10>',
                                                      'result'  : 'ap_int<10>'}}
    config['LayerName']['Dense_MatMul_1'] = {'Precision' : {'accum' : 'ap_int<10>',
                                                      'result'  : 'ap_int<10>'}}
    config['LayerName']['Dense_MatMul_2'] = {'Precision' : {'accum' : 'ap_int<10>',
                                                      'result'  : 'ap_int<10>'}}
    config['LayerName']['Dense_MatMul_3'] = {'Precision' : {'accum' : 'ap_int<10>',
                                                      'result'  : 'ap_int<10>'}}
    hls_model = hls4ml.converters.convert_from_onnx_model(model,
                                                          output_dir='hls4mlprj_qonnx_tfc-2w2a',
                                                          part='xcu250-figd2104-2L-e',
                                                          hls_config=config)
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)

if __name__ == '__main__':
    test_tfc_2w2a()
