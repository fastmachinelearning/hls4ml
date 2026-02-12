import os
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model, load_model

import hls4ml
import hls4ml.model

test_root_path = Path(__file__).parent

os.environ['XILINX_VITIS'] = "/tools/Xilinx/Vitis/2023.2"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']

XPFM_PATH = "/media/tanawin/tanawin1701e/project8/kv260plateform/xsaFile/vitis_design_wrapper_2.xsa"
LOG_STD = True


def create_io_file_dir():
    os.makedirs(test_root_path / "input_file", exist_ok=True)
    os.makedirs(test_root_path / "output_file", exist_ok=True)


def checkEqual(a, b):

    equal = np.array_equal(a, b)
    if equal:
        print("Test pass both are equal \U0001f642")
    else:
        print("Test Fail both are not equal \U0001f62c")
    return equal


def create_simple_testcase(inputShape=(4, 4, 1), fileName="inputX.npy"):
    n_in = np.random.rand(*inputShape).astype(np.float32)
    os.makedirs(test_root_path / "input_file", exist_ok=True)
    np.save(test_root_path / "input_file" / fileName, n_in)


def create_simple_unet(input_shape=(4, 4, 1), modelName="simpleSkip.keras"):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    # Bottleneck
    bn = Conv2D(4, (3, 3), activation='relu', padding='same')(p1)
    # Decoder
    u1 = UpSampling2D((2, 2))(bn)
    concat1 = Concatenate()([u1, c1])
    c2 = Conv2D(2, (3, 3), activation='relu', padding='same')(concat1)
    # Output layer (1 channel)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c2)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save(test_root_path / "input_file" / modelName)


def gen_prj_dir(backend, io_type, strategy, granularity, prefix, axi_mode):
    return str(test_root_path / f"hls4mlprj_{prefix}_{backend}_{strategy}_{io_type}_{granularity}_{axi_mode}")


def create_hls_model(model, config, backend, io_type, strategy, granularity, prefix, axi_mode):
    output_dir = gen_prj_dir(backend, io_type, strategy, granularity, prefix, axi_mode)
    # mono model build
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
        board='zcu102',
        part='xczu9eg-ffvb1156-2-e',
        clock_period='10ns',
        input_type="float",
        output_type="float",
        xpfmPath=XPFM_PATH,
        axi_mode=axi_mode,
    )
    hls_model.compile()
    return hls_model


def create_hls_model4_cosim(
    model, config, backend, io_type, strategy, granularity, input_data_tb, output_data_tb, prefix, axi_mode
):
    output_dir = gen_prj_dir(backend, io_type, strategy, granularity, prefix, axi_mode)
    # mono model build
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
        board='zcu102',
        part='xczu9eg-ffvb1156-2-e',
        clock_period='10ns',
        input_type="float",
        output_type="float",
        input_data_tb=input_data_tb,
        output_data_tb=output_data_tb,
        axi_mode=axi_mode,
    )
    hls_model.compile()
    return hls_model


def predict_hls_model(hls_model, input_data):
    y_hls4ml = hls_model.predict(input_data)
    return y_hls4ml


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis'])
def test_backend_predict(io_type, strategy, granularity, amt_query, axi_mode):
    create_io_file_dir()
    # create and load data set
    create_simple_testcase(inputShape=(amt_query, 4, 4, 1), fileName="inputX.npy")
    input_data = np.load(test_root_path / "input_file" / "inputX.npy")
    # create and load model
    model_name = "simpleSkip.keras"
    create_simple_unet(modelName=model_name)
    model = load_model(test_root_path / "input_file" / model_name)
    # config the keras model
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)

    # create hls4ml model
    vitis_unified_model = create_hls_model(model, config, "VitisUnified", io_type, strategy, granularity, "bridge", axi_mode)
    vitis_model = create_hls_model(model, config, "Vitis", io_type, strategy, granularity, "bridge", axi_mode)

    # predict test

    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    y_hls4ml = predict_hls_model(vitis_model, input_data)

    assert checkEqual(y_hls4ml_unified, y_hls4ml), "the result from vitis unified and vitis are not equal!"


# test_backend_predict("io_stream", 'latency', 'name', 10, "axis")


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis'])
def test_co_simulation(io_type, strategy, granularity, amt_query, axi_mode):
    create_io_file_dir()
    # create and load data set
    create_simple_testcase(inputShape=(amt_query, 4, 4, 1), fileName="inputCosim.npy")
    input_data = np.load(test_root_path / "input_file" / "inputCosim.npy")
    # create and load model
    model_name = "simpleSkipCosim.keras"
    create_simple_unet(modelName=model_name)
    model = load_model(test_root_path / "input_file" / model_name)
    # config the keras model
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, "VitisUnified", io_type, strategy, granularity, "precosim", axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(test_root_path / "output_file" / "outputCosim.npy", y_hls4ml_unified)

    input_data_tb = str(test_root_path / "input_file" / "inputCosim.npy")
    output_data_tb = str(test_root_path / "output_file" / "outputCosim.npy")

    # create hls4ml model
    vitis_unified_model_cosim = create_hls_model4_cosim(
        model, config, "VitisUnified", io_type, strategy, granularity, input_data_tb, output_data_tb, "cosim", axi_mode
    )
    # do cosim
    vitis_unified_model_cosim.compile()
    vitis_unified_model_cosim.build(synth=True, cosim=True, log_to_stdout=LOG_STD)

    bridge_result_path = (
        gen_prj_dir("VitisUnified", io_type, strategy, granularity, "cosim", axi_mode) + "/tb_data/tb_output_predictions.dat"
    )
    cosim_result_path = (
        gen_prj_dir("VitisUnified", io_type, strategy, granularity, "cosim", axi_mode) + "/tb_data/rtl_cosim_results.log"
    )

    bridge_result = np.loadtxt(bridge_result_path)
    cosim_result = np.loadtxt(cosim_result_path)

    assert np.allclose(bridge_result, cosim_result, rtol=0.0, atol=1e-4), "the result from bridge and cosim are not equal!"


# test_co_simulation("io_stream", 'latency', 'name', 10, 'axis')


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis'])
def test_fifo_depth(io_type, strategy, granularity, amt_query, axi_mode):
    create_io_file_dir()
    # create and load data set
    create_simple_testcase(inputShape=(amt_query, 4, 4, 1), fileName="inputFifoDepth.npy")
    input_data = np.load(test_root_path / "input_file" / "inputFifoDepth.npy")
    # create and load model
    model_name = "simpleSkipFifoDepth.keras"
    create_simple_unet(modelName=model_name)
    model = load_model(test_root_path / "input_file" / model_name)
    # config the keras model
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, "VitisUnified", io_type, strategy, granularity, "fifodepth", axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(test_root_path / "output_file" / "outputFifoDepth.npy", y_hls4ml_unified)

    input_data_tb = str(test_root_path / "input_file" / "inputFifoDepth.npy")
    output_data_tb = str(test_root_path / "output_file" / "outputFifoDepth.npy")

    # create hls4ml model
    config['Flows'] = ['vitisunified:fifo_depth_optimization']
    vitis_unified_model_fifo = create_hls_model4_cosim(
        model, config, "VitisUnified", io_type, strategy, granularity, input_data_tb, output_data_tb, "fifodepth", axi_mode
    )
    # do cosim
    vitis_unified_model_fifo.compile()

    fifodepth_result_path = (
        gen_prj_dir("VitisUnified", io_type, strategy, granularity, "fifodepth", axi_mode) + "/fifo_depths.json"
    )
    assert os.path.exists(fifodepth_result_path), "the fifo_depth file is not exist"


# test_fifo_depth("io_stream", 'latency', 'name', 10, "axis")


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10000])
@pytest.mark.parametrize('axi_mode', ['axis'])
def test_gen_unified(io_type, strategy, granularity, amt_query, axi_mode):
    create_io_file_dir()
    # create and load data set
    create_simple_testcase(inputShape=(amt_query, 4, 4, 1), fileName="inputGenbit.npy")
    input_data = np.load(test_root_path / "input_file" / "inputGenbit.npy")
    # create and load model
    model_name = "simpleSkipGenBit.keras"
    create_simple_unet(modelName=model_name)
    model = load_model(test_root_path / "input_file" / model_name)
    # config the keras model
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, "VitisUnified", io_type, strategy, granularity, "gen_unified", axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(test_root_path / "output_file" / "outputGenbit.npy", y_hls4ml_unified)

    vitis_unified_model.compile()
    vitis_unified_model.build(synth=True, bitfile=True, log_to_stdout=LOG_STD)


test_gen_unified("io_stream", 'latency', 'name', 10000, 'axis')
