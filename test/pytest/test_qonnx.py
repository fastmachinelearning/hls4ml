#!/usr/bin/env python
import pytest
import hls4ml
import numpy as np
import qonnx.util.cleanup
import qonnx.util.to_channels_last
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
    hls_model = hls4ml.converters.convert_from_onnx_model(model,
                                                          output_dir='hls4mlprj_qonnx_tfc-2w2a',
                                                          part='xcu250-figd2104-2L-e',
                                                          hls_config=config)
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)

def test_tfc_2w2a_quartus():
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
    config['LayerName']['global_in'] = {'Precision' : 'ac_fixed<16,2>'}
    hls_model = hls4ml.converters.convert_from_onnx_model(model,
                                                          output_dir='hls4mlprj_qonnx_tfc-2w2a-quartus',
                                                          part='Arria10',
                                                          backend='Quartus',
                                                          hls_config=config)
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)

def test_cnv_2w2a():
    # download test model
    dl_dir = "./"
    dl_file = dl_dir + "qonnx-cnv-2w2a.onnx"
    cnv_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
    )
    urllib.request.urlretrieve(cnv_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_clean = dl_dir + "/qonnx-cnv-2w2a-clean.onnx"
    out_chanlast = dl_dir + "/qonnx-cnv-2w2a-clean-channels-last.onnx"
    out_file = dl_dir + "/qonnx-cnv-2w2a-clean-channels-last-clean.onnx"

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_clean)
    qonnx.util.to_channels_last.to_channels_last(out_clean, make_input_channels_last=True, out_file=out_chanlast)
    qonnx.util.cleanup.cleanup(out_chanlast, out_file=out_file)
    model = ModelWrapper(out_file)

    # Execute QONNX model inference
    # TODO make the test bigger
    ishape = (1,32,32,3)
    np.random.seed(1)
    X = np.random.uniform(low=-1, high=+1, size=np.product(ishape)).reshape(ishape).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model)
    config['Model']['Precision'] = 'ap_fixed<32,16>'
    # Some hand-derived config
    # TODO should be auto-derived by QuantizeDenseOutput pass after some adaptation

    hls_model = hls4ml.converters.convert_from_onnx_model(model,
                                                          output_dir='hls4mlprj_qonnx_cnv-2w2a',
                                                          part='xcu250-figd2104-2L-e',
                                                          io_type='io_stream',
                                                          hls_config=config)
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)

@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_jet_tagging(backend):
    # download test model
    dl_dir = "./"
    dl_file = dl_dir + "qkeras_jettagging.onnx"
    jet_tagging_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/JetTagging/QKeras_hls4ml_3layer/qkeras_jettagging.onnx"
    )
    urllib.request.urlretrieve(jet_tagging_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = dl_dir + "/qkeras_jettagging-clean.onnx"

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)

    # Execute QONNX model inference
    # TODO make the test bigger
    ishape = (1,16)
    np.random.seed(0)
    X = np.random.uniform(low=-1, high=+1, size=np.product(ishape)).reshape(ishape).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model)
    # Some hand-derived config
    # TODO should be auto-derived by QuantizeDenseOutput pass after some adaptation

    hls_model = hls4ml.converters.convert_from_onnx_model(model,
                                                          output_dir=f'hls4mlprj_qonnx_jettag_{backend}',
                                                          backend=backend,
                                                          hls_config=config)
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


if __name__ == '__main__':
    test_tfc_2w2a_quartus()
