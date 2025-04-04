import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../../example-models').resolve()

backend_options = ['VitisAcceleratorIPFlow']


def parse_cosim_report_and_search_for_bitstream(project_path):
    """Parse the cosimulation report to check whether the cosimulation passed or failed and therefore a deadlock is
    detected and check if the bitstream was generated without errors.
    """
    prj_dir = None
    top_func_name = None

    project_tcl_path = project_path + '/project.tcl'

    with open(project_tcl_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'

    cosim_file_path = project_path + '/' + prj_dir + f'/solution1/sim/report/{top_func_name}_axi_cosim.rpt'
    bitsteam_path = (
        project_path + '/' + f"{top_func_name}_vitis_accelerator_ip_flow/project_1.runs/impl_1/design_1_wrapper.bit"
    )

    cosim_report_exists = os.path.isfile(cosim_file_path)
    bitstream_exists = os.path.isfile(bitsteam_path)

    if cosim_report_exists and bitstream_exists:
        return cosim_file_path, bitstream_exists
    elif (not cosim_report_exists) and (not bitstream_exists):
        raise FileNotFoundError("Co-simulation report and Bitstream not found.")
    elif not cosim_report_exists:
        raise FileNotFoundError("Co-simulation report not found.")
    else:
        raise FileNotFoundError("Bitstream not found.")


def run_bitstream_generation_keras(backend, profiling_fifo_depth, io_type, run_fifo_depth_optimization):
    """Execute bitstream generation on a dummy Keras model and the FIFO optimization sequence if
    `run_fifo_depth_optimization` is set.
    """

    # create a keras model
    input_shape = (16, 16, 3)
    activation = 'relu'
    kernel_size = (3, 3)
    padding = 'same'

    model = Sequential()
    model.add(
        SeparableConv2D(filters=4, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape)
    )
    model.add(SeparableConv2D(filters=8, kernel_size=kernel_size, padding=padding, activation=activation))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(1, *input_shape)
    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32, 16>')

    # include the FIFO Depth optimizer do the flows if `run_fifo_depth_optimization` is set
    if run_fifo_depth_optimization:
        config['Flows'] = ['vitisacceleratoripflow:fifo_depth_optimization']
        hls4ml.model.optimizer.get_optimizer('vitisacceleratoripflow:fifo_depth_optimization').configure(
            profiling_fifo_depth=profiling_fifo_depth
        )

    output_dir = str(
        test_root_path / f'hls4mlprj_keras_model_backend_{backend}_fifo_optimization_{run_fifo_depth_optimization}'
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type=io_type, hls_config=config, output_dir=output_dir, backend=backend, clock_period=10
    )

    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.01)

    # build the hls4ml model and check if the bitstream was generated and the FIFOs were optimized if
    # `run_fifo_depth_optimization` is set
    build_and_check(hls_model, run_fifo_depth_optimization)


def build_and_check(hls_model, run_fifo_depth_optimization):
    """Execute the FIFO depth optimization sequence on an hls4ml model."""

    # try to generate a bitstream. Use the optimized FIFO depths and execute cosimulation to check for deadlocks
    # due to the new FIFO depths if `run_fifo_depth_optimization` is set
    hls_model.build(
        reset=False, synth=True, csim=False, export=True, cosim=True, bitfile=True, fifo_opt=run_fifo_depth_optimization
    )

    # checks if the fifo depths decreased/were optimized
    fifo_depths_decreased = False
    if run_fifo_depth_optimization:
        fifo_depths = {}
        with open(hls_model.config.get_output_dir() + "/fifo_depths.json") as fifo_depths_file:
            fifo_depths = json.load(fifo_depths_file)

        # omit checking for the input and output AXIS FIFOs as they are not always optimized
        # as the last kernel e.g pointwise is faster than AXIS speed
        fifo_depths_decreased = all(
            fifo_depths['optimized'] < fifo_depths['initial']
            for fifo_name, fifo_depths in fifo_depths.items()
            if fifo_name not in {'in_local', 'out_local'}
        )

    # checks that the cosimulation ran succesfully without detecting deadlocks and if the bitstream was generated
    cosim_report_path, bitstream_exists = parse_cosim_report_and_search_for_bitstream(hls_model.config.get_output_dir())

    with open(cosim_report_path) as cosim_report_file:
        cosim_succesful = any("Pass" in line for line in cosim_report_file)

    assert (fifo_depths_decreased or (not run_fifo_depth_optimization)) and cosim_succesful and bitstream_exists


def expect_exception(error, message, backend, profiling_fifo_depth, io_type):
    with pytest.raises(error, match=re.escape(message)):
        run_bitstream_generation_keras(backend, profiling_fifo_depth, io_type)


def get_branched_model():
    """
    Load branched model, already channels-last and cleaned.
    """
    dl_file = str(example_model_path / "onnx/branched_model_ch_last.onnx")
    assert os.path.isfile(dl_file)
    model = ModelWrapper(dl_file)
    return model


def run_bitstream_generation_onnx(backend, profiling_fifo_depth, io_type, model, run_fifo_depth_optimization):
    """Execute bitstream generation on a QONNX branched model and the FIFO optimization sequence if
    `run_fifo_depth_optimization` is set.
    """

    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend=backend, default_precision='ap_fixed<15,2,AP_RND_CONV>'
    )

    # add this line to remove the linear layer that quantizes the input of the NN
    config['LayerName']['global_in']['Precision']['result'] = 'fixed<4,0,AP_RND_CONV,AP_SAT,0>'

    if run_fifo_depth_optimization:
        config['Flows'] = ['vitisacceleratoripflow:fifo_depth_optimization']
        hls4ml.model.optimizer.get_optimizer('vitisacceleratoripflow:fifo_depth_optimization').configure(
            profiling_fifo_depth=profiling_fifo_depth
        )

    output_dir = str(
        test_root_path / f'hls4mlprj_branched_model_backend_{backend}_fifo_optimization_{run_fifo_depth_optimization}'
    )

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=output_dir,
        io_type=io_type,
        backend=backend,
        hls_config=config,
        part="xczu9eg-ffvb1156-2-e",
        board='zcu102',
        clock_period=10,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))
    np.testing.assert_array_equal(y_qonnx.ravel(), y_hls4ml.ravel())

    build_and_check(hls_model, run_fifo_depth_optimization)


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
@pytest.mark.parametrize('profiling_fifo_depth', [-2, 3.14, "a"])
def test_value_error(backend, profiling_fifo_depth):
    """Test the FIFO depth optimizer with faulty inputs of profiling_fifo_depth to verify that an exception is raised."""
    message = "The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer."
    expect_exception(
        ValueError, message, backend, profiling_fifo_depth, io_type='io_stream', run_fifo_depth_optimization=True
    )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_runtime_error(backend):
    """Test the FIFO depth optimizer with io_type='io_parallel' to verify that an exception is raised."""
    message = "To use this optimization you have to set `IOType` field to `io_stream` in the HLS config."
    expect_exception(
        RuntimeError, message, backend, profiling_fifo_depth=200_000, io_type='io_parallel', run_fifo_depth_optimization=True
    )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_dummy_keras(backend):
    """Test the correct execution of the bitstream generation."""
    run_bitstream_generation_keras(
        backend, profiling_fifo_depth=200_000, io_type='io_stream', run_fifo_depth_optimization=False
    )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_dummy_keras_with_fifo_optimization(backend):
    """Test the correct execution of the bitstream generation with the FIFO depth optimizer."""
    run_bitstream_generation_keras(
        backend, profiling_fifo_depth=200_000, io_type='io_stream', run_fifo_depth_optimization=True
    )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_branched_model(backend):
    """Test the correct execution of the bitstream generation."""
    run_bitstream_generation_onnx(
        backend,
        profiling_fifo_depth=200_000,
        io_type='io_stream',
        model=get_branched_model(),
        run_fifo_depth_optimization=False,
    )


@pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_successful_execution_of_branched_model_with_fifo_optimization(backend):
    """Test the correct execution of the bitstream generation with the FIFO depth optimizer."""
    run_bitstream_generation_onnx(
        backend,
        profiling_fifo_depth=200_000,
        io_type='io_stream',
        model=get_branched_model(),
        run_fifo_depth_optimization=True,
    )
