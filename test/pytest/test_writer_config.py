import os
import shutil
from pathlib import Path

import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def keras_model():
    model = Sequential()
    model.add(Dense(10, kernel_initializer='zeros', use_bias=False, input_shape=(15,)))
    model.compile()
    return model


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])  # No Quartus for now
@pytest.mark.parametrize('namespace', [None, 'test_namespace'])
def test_namespace(keras_model, namespace, io_type, backend):

    use_namespace = namespace is None
    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    odir = str(test_root_path / f'hls4mlprj_namespace_{use_namespace}_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=config, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()  # It's enough that the model compiles


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])  # No Quartus for now
@pytest.mark.parametrize('write_tar', [True, False])
def test_write_tar(keras_model, write_tar, backend):

    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    odir = str(test_root_path / f'hls4mlprj_write_tar_{write_tar}_{backend}')

    if os.path.exists(odir + '.tar.gz'):
        os.remove(odir + '.tar.gz')

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=config, output_dir=odir, backend=backend, write_tar=write_tar
    )
    hls_model.write()

    tar_written = os.path.exists(odir + '.tar.gz')
    assert tar_written == write_tar


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])  # No Quartus for now
@pytest.mark.parametrize('write_weights_txt', [True, False])
def test_write_weights_txt(keras_model, write_weights_txt, backend):

    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    odir = str(test_root_path / f'hls4mlprj_write_weights_txt_{write_weights_txt}_{backend}')

    if os.path.exists(odir):
        shutil.rmtree(odir)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=config, output_dir=odir, backend=backend, write_weights_txt=write_weights_txt
    )
    hls_model.write()

    txt_written = os.path.exists(odir + '/firmware/weights/w2.txt')
    assert txt_written == write_weights_txt


@pytest.mark.skip(reason='Skipping for now as it needs the installation of the compiler.')
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('tb_output_stream', ['stdout', 'file', 'both'])
def test_tb_output_stream(capfd, keras_model, tb_output_stream, backend):

    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    odir = str(test_root_path / f'hls4mlprj_tb_output_stream_{tb_output_stream}_{backend}')
    if os.path.exists(odir):
        shutil.rmtree(odir)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        io_type='io_stream',
        hls_config=config,
        output_dir=odir,
        backend=backend,
        tb_output_stream=tb_output_stream,
    )
    hls_model.build(csim=True, synth=False)

    # Check the output based on tb_output_stream
    tb_file_path = os.path.join(odir, 'tb_data/csim_results.log')

    with open(tb_file_path) as tb_file:
        tb_content = tb_file.read()
        if tb_output_stream in ['file', 'both']:
            assert len(tb_content) > 0, 'Testbench output file expected to contain model outputs, but is empty'
        else:
            assert len(tb_content) == 0, 'Testbench output file expected to be empty, but contains data'

    captured = capfd.readouterr()
    if tb_output_stream in ['stdout', 'both']:
        assert '0 0 0 0 0 0 0 0 0 0' in captured.out, 'Expected model output not found in stdout'
    else:
        assert '0 0 0 0 0 0 0 0 0 0' not in captured.out, 'Model output should not be printed to stdout'
