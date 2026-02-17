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

test_root_path = Path(__file__).parent
INPUT_DIR = test_root_path / 'input_file'
OUTPUT_DIR = test_root_path / 'output_file'


# vendor = AMD
XPFM_PATH = '/tools/Xilinx/Vitis/2023.2/base_platforms/xilinx_zcu102_base_202320_1/xilinx_zcu102_base_202320_1.xpfm'


LOG_STD = True


def create_io_file_dir():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_simple_testcase(input_shape=(4, 4, 1), file_name='X.npy'):
    n_in = np.random.rand(*input_shape).astype(np.float32)
    os.makedirs(INPUT_DIR, exist_ok=True)
    np.save(INPUT_DIR / file_name, n_in)


def init_simple_unet(input_shape=(4, 4, 1), model_name='simple_skip.keras'):
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
    model.save(INPUT_DIR / model_name)


def gen_prj_dir(backend, io_type, strategy, granularity, prefix, axi_mode):
    return str(test_root_path / f'hls4mlprj_{prefix}_{backend}_{strategy}_{io_type}_{granularity}_{axi_mode}')


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
        input_type='float',
        output_type='float',
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
        input_type='float',
        output_type='float',
        input_data_tb=input_data_tb,
        output_data_tb=output_data_tb,
        axi_mode=axi_mode,
    )
    hls_model.compile()
    return hls_model


def predict_hls_model(hls_model, input_data):
    y_hls4ml = hls_model.predict(input_data)
    return y_hls4ml


def prepare_test_case(amt_query, input_file, model_file, granularity):
    create_io_file_dir()
    init_simple_testcase(input_shape=(amt_query, 4, 4, 1), file_name=input_file)
    input_data = np.load(INPUT_DIR / input_file)
    init_simple_unet(model_name=model_file)
    model = load_model(INPUT_DIR / model_file)
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    return input_data, model, config


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis', 'axim'])
def test_backend_predict(io_type, strategy, granularity, amt_query, axi_mode):
    input_data, model, config = prepare_test_case(
        amt_query=amt_query, input_file='X.npy', model_file='simple_skip.keras', granularity=granularity
    )

    # create hls4ml model
    vitis_unified_model = create_hls_model(model, config, 'VitisUnified', io_type, strategy, granularity, 'bridge', axi_mode)
    vitis_model = create_hls_model(model, config, 'Vitis', io_type, strategy, granularity, 'bridge', axi_mode)

    # predict test

    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    y_hls4ml = predict_hls_model(vitis_model, input_data)

    np.testing.assert_array_equal(y_hls4ml_unified, y_hls4ml)


# test_backend_predict("io_stream", 'latency', 'name', 10, "axim")


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis', 'axim'])
def test_co_simulation(io_type, strategy, granularity, amt_query, axi_mode):
    input_data, model, config = prepare_test_case(
        amt_query=amt_query, input_file='cosim_X.npy', model_file='cosim_simple_skip.keras', granularity=granularity
    )

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, 'VitisUnified', io_type, strategy, granularity, 'precosim', axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(OUTPUT_DIR / 'YCosim.npy', y_hls4ml_unified)

    input_data_tb = str(INPUT_DIR / 'cosim_X.npy')
    output_data_tb = str(OUTPUT_DIR / 'cosim_Y.npy')

    # create hls4ml model
    vitis_unified_model_cosim = create_hls_model4_cosim(
        model, config, 'VitisUnified', io_type, strategy, granularity, input_data_tb, output_data_tb, 'cosim', axi_mode
    )
    # do cosim
    vitis_unified_model_cosim.compile()
    vitis_unified_model_cosim.build(synth=True, cosim=True, log_to_stdout=LOG_STD)

    bridge_result_path = (
        gen_prj_dir('VitisUnified', io_type, strategy, granularity, 'cosim', axi_mode) + '/tb_data/tb_output_predictions.dat'
    )
    cosim_result_path = (
        gen_prj_dir('VitisUnified', io_type, strategy, granularity, 'cosim', axi_mode) + '/tb_data/rtl_cosim_results.log'
    )

    bridge_result = np.loadtxt(bridge_result_path)
    cosim_result = np.loadtxt(cosim_result_path)

    assert np.allclose(bridge_result, cosim_result, rtol=0.0, atol=1e-4), 'the result from bridge and cosim are not equal!'


# test_co_simulation("io_stream", 'latency', 'name', 10, 'axim')


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis', 'axim'])
def test_csim_simulation(io_type, strategy, granularity, amt_query, axi_mode):
    input_data, model, config = prepare_test_case(
        amt_query=amt_query, input_file='csim_X.npy', model_file='csim_simple_skip.keras', granularity=granularity
    )
    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, 'VitisUnified', io_type, strategy, granularity, 'precsim', axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(OUTPUT_DIR / 'csim_Y.npy', y_hls4ml_unified)

    input_data_tb = str(INPUT_DIR / 'csim_X.npy')
    output_data_tb = str(OUTPUT_DIR / 'csim_Y.npy')

    # create hls4ml model
    vitis_unified_model_cosim = create_hls_model4_cosim(
        model, config, 'VitisUnified', io_type, strategy, granularity, input_data_tb, output_data_tb, 'csim', axi_mode
    )
    # do csim
    vitis_unified_model_cosim.compile()
    vitis_unified_model_cosim.build(synth=True, csim=True, log_to_stdout=LOG_STD)

    bridge_result_path = (
        gen_prj_dir('VitisUnified', io_type, strategy, granularity, 'csim', axi_mode) + '/tb_data/tb_output_predictions.dat'
    )
    cosim_result_path = (
        gen_prj_dir('VitisUnified', io_type, strategy, granularity, 'csim', axi_mode) + '/tb_data/csim_results.log'
    )

    bridge_result = np.loadtxt(bridge_result_path)
    cosim_result = np.loadtxt(cosim_result_path)

    assert np.allclose(bridge_result, cosim_result, rtol=0.0, atol=1e-4), 'the result from bridge and cosim are not equal!'


# test_csim_simulation("io_stream", 'latency', 'name', 10, 'axim')


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10])
@pytest.mark.parametrize('axi_mode', ['axis', 'axim'])
def test_fifo_depth(io_type, strategy, granularity, amt_query, axi_mode):
    input_data, model, config = prepare_test_case(
        amt_query=amt_query,
        input_file='fifo_depth_X.npy',
        model_file='fifo_depth_simple_skip.keras',
        granularity=granularity,
    )

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, 'VitisUnified', io_type, strategy, granularity, 'fifodepth', axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(OUTPUT_DIR / 'fifo_depth_Y.npy', y_hls4ml_unified)

    input_data_tb = str(INPUT_DIR / 'fifo_depth_X.npy')
    output_data_tb = str(OUTPUT_DIR / 'fifo_depth_Y.npy')

    # create hls4ml model
    config['Flows'] = ['vitisunified:fifo_depth_optimization']
    vitis_unified_model_fifo = create_hls_model4_cosim(
        model, config, 'VitisUnified', io_type, strategy, granularity, input_data_tb, output_data_tb, 'fifodepth', axi_mode
    )
    # do cosim
    vitis_unified_model_fifo.compile()

    fifodepth_result_path = (
        gen_prj_dir('VitisUnified', io_type, strategy, granularity, 'fifodepth', axi_mode) + '/fifo_depths.json'
    )
    assert os.path.exists(fifodepth_result_path), 'the fifo_depth file is not exist'


# test_fifo_depth("io_stream", 'latency', 'name', 10, "axim")


@pytest.mark.parametrize('io_type', ['io_stream'])
@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('granularity', ['name'])
@pytest.mark.parametrize('amt_query', [10000])
@pytest.mark.parametrize('axi_mode', ['axis', 'axim'])
def test_gen_unified(io_type, strategy, granularity, amt_query, axi_mode):
    input_data, model, config = prepare_test_case(
        amt_query=amt_query, input_file='gen_bit_X.npy', model_file='gen_bit_simple_skip.keras', granularity=granularity
    )

    # predict it first
    vitis_unified_model = create_hls_model(
        model, config, 'VitisUnified', io_type, strategy, granularity, 'gen_unified', axi_mode
    )
    y_hls4ml_unified = predict_hls_model(vitis_unified_model, input_data)
    np.save(OUTPUT_DIR / 'gen_bit_Y.npy', y_hls4ml_unified)

    vitis_unified_model.compile()
    vitis_unified_model.build(synth=True, bitfile=True, log_to_stdout=LOG_STD)


# test_gen_unified("io_stream", 'latency', 'name', 10000, 'axis')
