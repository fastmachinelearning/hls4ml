from pathlib import Path

import pytest
import tensorflow as tf

import hls4ml
from hls4ml.backends.altera.altera_backend import AlteraBackend
from hls4ml.backends.oneapi.oneapi_backend import OneAPIBackend
from hls4ml.model.flow import get_flow
from hls4ml.model.optimizer import get_optimizer
from hls4ml.writer.altera_writer import AlteraWriter
from hls4ml.writer.oneapi_writer import OneAPIWriter


def test_oneapi_backend_registration():
    altera = hls4ml.backends.get_backend('Altera')
    oneapi = hls4ml.backends.get_backend('oneAPI')

    assert type(altera) is AlteraBackend
    assert type(oneapi) is OneAPIBackend
    assert isinstance(oneapi, AlteraBackend)
    assert type(altera.writer) is AlteraWriter
    assert type(oneapi.writer) is OneAPIWriter

    for backend_name in ('altera', 'oneapi'):
        assert get_optimizer(f'{backend_name}:transform_types') is not None
        assert get_optimizer(f'{backend_name}:write_hls') is not None
        assert get_flow(f'{backend_name}:ip') is not None
        assert get_flow(f'{backend_name}:write').requires == [f'{backend_name}:ip']


@pytest.mark.parametrize(
    'backend,compiler,definition',
    [('Altera', 'ahls', None), ('oneAPI', 'icpx', 'HLS4ML_ONEAPI')],
)
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_altera_family_generated_project(tmp_path, backend, compiler, definition, io_type):
    keras_model = tf.keras.Sequential([tf.keras.Input(shape=(4,)), tf.keras.layers.Dense(2)])
    config = hls4ml.utils.config_from_keras_model(keras_model, backend=backend)
    output_dir = tmp_path / f'{backend}-{io_type}'

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=str(output_dir),
        backend=backend,
        io_type=io_type,
    )
    hls_model.write()

    cmake = (output_dir / 'CMakeLists.txt').read_text()
    assert f'set(CMAKE_CXX_COMPILER {compiler})' in cmake
    if definition is None:
        assert 'add_compile_definitions(HLS4ML_ONEAPI)' not in cmake
    else:
        assert f'add_compile_definitions({definition})' in cmake

    compatibility_header = (output_dir / 'src/firmware/nnet_utils/hls4ml_sycl.h').read_text()
    assert '#ifdef HLS4ML_ONEAPI' in compatibility_header
    assert 'namespace hls4ml_sycl_ext = sycl::ext::intel;' in compatibility_header
    assert 'namespace hls4ml_sycl_ext = sycl::ext::altera;' in compatibility_header

    generated_sources = [
        output_dir / 'src/firmware/myproject.h',
        output_dir / 'src/firmware/myproject.cpp',
        output_dir / 'src/myproject_test.cpp',
        output_dir / 'src/myproject_bridge.cpp',
    ]
    generated_text = '\n'.join(path.read_text() for path in generated_sources)
    assert 'hls4ml_sycl_ext::' in generated_text
    assert 'hls4ml_sycl_ext::experimental::pipe<' in generated_text
    assert 'sycl::ext::altera::' not in generated_text
    assert 'sycl::ext::intel::' not in generated_text

    assert Path(output_dir / 'src/firmware/nnet_utils/nnet_common.h').exists()
