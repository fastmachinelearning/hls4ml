import os
from pathlib import Path

import numpy as np
import pytest
from pquant.activations import PQActivation
from pquant.core.finetuning import TuningConfig
from pquant.core.utils import get_default_config
from pquant.layers import PQAvgPool1d, PQAvgPool2d, PQBatchNormalization, PQConv1d, PQConv2d, PQDense

from hls4ml.converters import convert_from_keras_model
from hls4ml.utils import config_from_keras_model

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras  # noqa: E402

test_path = Path(__file__).parent


def _run_synth_match_test(PQmodel: keras.Model, data, io_type: str, backend: str, dir: str, cond=None, strategy='latency'):
    output_dir = dir + '/hls4ml_prj'
    hls_config = config_from_keras_model(
        PQmodel,
        granularity='name',
        default_precision='ap_fixed<32, 16>',
        backend=backend,
    )
    hls_model = convert_from_keras_model(
        PQmodel,
        io_type=io_type,
        output_dir=output_dir,
        backend=backend,
        hls_config=hls_config,
    )
    hls_model.compile()

    data_len = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
    r_pq: list[np.ndarray] = [PQmodel(data).numpy()]  # type: ignore
    r_hls: list[np.ndarray] = [hls_model.predict(np.ascontiguousarray(data)).reshape(r_pq[0].shape)]  # type: ignore

    errors = []
    for i, (p, h) in enumerate(zip(r_pq, r_hls)):
        try:
            if cond is None:
                mismatch_ph = p != h
                assert np.sum(mismatch_ph) == 0, (
                    f'Proxy-HLS4ML mismatch for out {i}: {np.sum(np.any(mismatch_ph, axis=1))} out of {data_len} samples are different. Sample: {p[mismatch_ph].ravel()[:5]} vs {h[mismatch_ph].ravel()[:5]}'  # noqa: E501
                )
            else:
                cond(p, h)
        except AssertionError as e:
            errors.append(e)
    if len(errors) > 0:
        msgs = [str(e) for e in errors]
        raise AssertionError('\n'.join(msgs))


def run_model_test(
    PQmodel: keras.Model,
    data,
    io_type: str,
    backend: str,
    dir: str,
    cond=None,
    strategy='latency',
):
    _run_synth_match_test(PQmodel, data, io_type, backend, dir, cond=cond, strategy=strategy)


def create_pqlayer_model(layer: str, use_hgq: bool):
    config = get_default_config('pdp')
    config['pruning_parameters']['disable_pruning_for_layers'] = ['']
    config['quantization_parameters']['use_high_granularity_quantization'] = use_hgq
    config = TuningConfig.load_from_config(config)

    idx = layer.find('(') + 1
    layer = (
        layer[:idx]
        + 'config, '
        + layer[idx:-1]
        + (', quantize_output=True, out_quant_bits=(1., 2., 7.)' if 'BatchNorm' not in layer else '')
        + ')'
    )
    _layer = eval(layer)

    shape = get_shape(_layer)
    inp = keras.Input(shape[1:])
    out = _layer(inp)
    if 'BatchNorm' in layer:
        flat = keras.layers.Flatten()
        _layer2 = PQDense(config, 16, in_quant_bits=(1.0, 1.0, 7.0), quantize_output=True, out_quant_bits=(1.0, 2.0, 7.0))
        out = _layer2(flat(out))
    model = keras.Model(inp, out)

    return model, shape


def get_data(shape: tuple[int, ...], v: float, max_scale: float):
    rng = np.random.default_rng()
    a1 = rng.uniform(-v, v, shape).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, *shape[1:])).astype(np.float32)
    return (a1 * a2).astype(np.float32)


def get_shape(
    layer: keras.layers.Layer,
    batch_size: int = 1,
    default_length: int = 32,
    default_hw: tuple[int, int] = (32, 32),
    default_channels: int = 2,
):
    match layer:
        case PQActivation():
            # (N, L)
            return (batch_size, default_length)
        case PQAvgPool1d():
            # (N, L, C)
            return (batch_size, default_length, default_channels)
        case PQAvgPool2d():
            # (N, H, W, C)
            return (batch_size, *default_hw, default_channels)
        case PQBatchNormalization():
            # (N, num_features, H, W)
            return (batch_size, *default_hw, default_channels)
        case PQConv1d():
            # (N, C_in, L)
            return (batch_size, default_length, default_channels)
        case PQConv2d():
            # (N, C_in, H, W)
            return (batch_size, *default_hw, default_channels)
        case PQDense():
            # (N, in_features)
            return (batch_size, default_length)
        case _:
            raise TypeError(f'Unsupported layer type: {type(layer).__name__}')


@pytest.mark.parametrize(
    'layer',
    [
        'PQDense(16)',
        'PQDense(16, use_bias=False)',
        "PQConv1d(3, kernel_size=3, padding='same')",
        "PQConv1d(3, kernel_size=3, padding='valid')",
        "PQConv1d(3, kernel_size=3, padding='valid', use_bias=False)",
        "PQConv1d(3, kernel_size=3, padding='valid', strides=2)",
        "PQConv1d(3, kernel_size=3, padding='same', strides=2)",
        "PQConv2d(3, kernel_size=(3,3), padding='same')",
        "PQConv2d(3, kernel_size=(3,3), padding='valid')",
        "PQConv2d(3, kernel_size=(3,3), padding='valid', use_bias=False)",
        "PQConv2d(3, kernel_size=(3,3), padding='valid', strides=2)",
        "PQConv2d(3, kernel_size=(3,3), padding='same', strides=2)",
        'PQBatchNormalization()',
        "PQAvgPool1d(2, padding='same')",
        "PQAvgPool2d((1,2), padding='same')",
        "PQAvgPool2d((2,2), padding='same')",
        "PQAvgPool1d(2, padding='valid')",
        "PQAvgPool2d((1,2), padding='valid')",
        "PQAvgPool2d((2,2), padding='valid')PQActivation('relu')",
        "PQActivation('tanh')",
    ],
)
@pytest.mark.parametrize('N', [1000])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['vivado', 'vitis'])
@pytest.mark.parametrize('use_hgq', [True, False])
@pytest.mark.parametrize('strategy', ['latency', 'resource'])
def test_syn_hlayers(layer, N: int, io_type: str, backend: str, use_hgq: bool, strategy: str):
    model, data_shape = create_pqlayer_model(layer=layer, use_hgq=use_hgq)
    data = get_data(data_shape, 7, 1)

    path = test_path / f'hls4mlprj_pquant_keras__{layer}_{io_type}_{backend}_{use_hgq}_{strategy}'

    run_model_test(model, data, io_type, backend, str(path), None, strategy)
