from pathlib import Path

import numpy as np
import pytest
from qkeras.qconvolutional import QConv2D
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import binary, quantized_bits, quantized_relu, ternary
from qonnx.core.modelwrapper import ModelWrapper
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()


def qkeras_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        QConv2D(4, kernel_size=(3, 3), kernel_quantizer='quantized_bits(8,4)', bias_quantizer=ternary(), name='conv2d_1')
    )
    model.add(QActivation(name='relu_1', activation='quantized_relu(6)'))
    model.add(
        QConv2D(5, kernel_size=(3, 3), kernel_quantizer=quantized_bits(8, 4), bias_quantizer=binary(), name='conv2d_2')
    )
    model.add(QActivation(name='relu_2', activation=quantized_relu(2)))
    model.add(Flatten())
    model.add(QDense(5, activation='softmax', kernel_quantizer='quantized_po2(8,4)', name='dense_1'))
    return model


@pytest.mark.parametrize('backend', ['Vitis', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_save_load__model(io_type, backend):
    input_shape = (8, 8, 3)

    keras_model = qkeras_model(input_shape)

    X = np.random.uniform(low=0, high=1, size=10 * np.prod(input_shape)).reshape((10, *input_shape))
    X = (np.round(X * 2**10) * 2**-10).astype(np.float32)

    config = hls4ml.utils.config.config_from_keras_model(
        keras_model, granularity='name', backend=backend, default_precision='fixed<16,6>'
    )

    for layer in config['LayerName']:
        if layer.startswith('Softmax'):
            config['LayerName'][layer]['Implementation'] = 'legacy'

    out_dir = test_root_path / f'hls4mlprj_serialization_qkeras_{io_type}_{backend}'

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        output_dir=str(out_dir / 'original'),
        io_type=io_type,
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_original = hls_model.predict(X)

    hls_model.save(out_dir / 'qonnx_model.fml')
    hls_model_clone = hls4ml.converters.load_saved_model(out_dir / 'qonnx_model.fml')
    hls_model_clone.config.config['OutputDir'] = str(out_dir / 'clone')
    hls_model_clone.compile()
    y_clone = hls_model_clone.predict(X)

    np.testing.assert_equal(y_original, y_clone)


@pytest.mark.parametrize('backend', ['Vitis'])  # Disabling OneAPI for now excessive run time
def test_save_load_qonnx_model(backend):
    dl_file = str(example_model_path / 'onnx/branched_model_ch_last.onnx')

    qonnx_model = ModelWrapper(dl_file)

    ishape = tuple(qonnx_model.get_tensor_shape(qonnx_model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**10) * 2**-10).astype(np.float32)

    config = hls4ml.utils.config.config_from_onnx_model(
        qonnx_model, granularity='name', backend=backend, default_precision='fixed<16,6>'
    )

    for layer in config['LayerName']:
        if layer.startswith('Softmax'):
            config['LayerName'][layer]['Implementation'] = 'legacy'

    out_dir = test_root_path / f'hls4mlprj_serialization_onnx_{backend}'

    hls_model = hls4ml.converters.convert_from_onnx_model(
        qonnx_model,
        output_dir=str(out_dir / 'original'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_original = hls_model.predict(X)

    hls_model.save(out_dir / 'qonnx_model.fml')
    hls_model_clone = hls4ml.converters.load_saved_model(out_dir / 'qonnx_model.fml')
    hls_model_clone.config.config['OutputDir'] = str(out_dir / 'clone')
    hls_model_clone.compile()
    y_clone = hls_model_clone.predict(X)

    np.testing.assert_equal(y_original, y_clone)


@pytest.mark.parametrize('backend', ['Vitis', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linking_project(io_type, backend):
    input_shape = (8, 8, 3)

    keras_model = qkeras_model(input_shape)

    X = np.random.uniform(low=0, high=1, size=10 * np.prod(input_shape)).reshape((10, *input_shape))
    X = (np.round(X * 2**10) * 2**-10).astype(np.float32)

    config = hls4ml.utils.config.config_from_keras_model(
        keras_model, granularity='name', backend=backend, default_precision='fixed<16,6>'
    )

    for layer in config['LayerName']:
        if layer.startswith('Softmax'):
            config['LayerName'][layer]['Implementation'] = 'legacy'

    out_dir = test_root_path / f'hls4mlprj_link_project_{io_type}_{backend}'

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        output_dir=str(out_dir),
        io_type=io_type,
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_original = hls_model.predict(X)

    hls_model_clone = hls4ml.converters.link_existing_project(out_dir)
    hls_model_clone.compile()
    y_clone = hls_model_clone.predict(X)

    np.testing.assert_equal(y_original, y_clone)
