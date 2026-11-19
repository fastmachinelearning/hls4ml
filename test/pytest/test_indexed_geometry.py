from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import hls4ml
from hls4ml.contrib.indexed_geometry import (
    IndexedConvolutionLayer,
    IndexedPoolingLayer,
    NeighborGatherLayer,
)

test_root_path = Path(__file__).parent

N_PIXELS = 16
N_NEIGHBORS = 7
N_FEATURES = 4
N_FILT_1 = 4
N_FILT_2 = 8


def _circular_neighbor_map(n_pixels=N_PIXELS, n_neighbors=N_NEIGHBORS, with_border=False):
    """
    Synthetic neighbor map: a circular shift, so every pixel has n_neighbors
    valid neighbors by construction. With with_border=True, the last two
    neighbor slots of the last three pixels are set to -1 (border pixels),
    to exercise the zero-masking path.
    """
    indices = np.stack(
        [np.roll(np.arange(n_pixels), -k) for k in range(1, n_neighbors + 1)],
        axis=1,
    ).astype(np.int32)
    if with_border:
        indices[-3:, -2:] = -1
    return indices


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_neighbor_gather(test_case_id, io_type):
    """
    NeighborGatherLayer, unpacked mode, chained into a native Conv2D probe
    layer, matching test_neighbor_gather_2d.py / test_neighbor_gather_2d_stream.py.
    """
    backend_id = 'Vitis'
    neighbor_indices = _circular_neighbor_map()

    inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_1')(inp)
    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), padding='same', activation='relu', name='probe_conv')(x)
    kmodel = tf.keras.Model(inputs=inp, outputs=x)

    hls_config = hls4ml.utils.config_from_keras_model(
        kmodel, granularity='name', backend=backend_id, default_precision='ap_fixed<16,6>', default_reuse_factor=1
    )
    output_dir = str(test_root_path / test_case_id)
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel, output_dir=output_dir, backend=backend_id, io_type=io_type, hls_config=hls_config
    )
    hmodel.compile()

    rng = np.random.default_rng(42)
    x_test = rng.standard_normal((4, N_PIXELS, N_FEATURES)).astype(np.float32)
    y_keras = kmodel.predict(x_test, verbose=0)
    y_hls = hmodel.predict(x_test).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=0.1, verbose=True)


def test_neighbor_gather_packed_shape(test_case_id):
    """
    NeighborGatherLayer with pack_neighbors=True (io_stream only): verifies
    the PackNeighbors flag propagates and the generated output type is
    flattened to n_neighbors*n_features, matching the packed-mode contract.
    No downstream layer is attached, since a generic consumer cannot accept
    a packed output (see indexed_conv/indexed_pool for packed-aware
    consumers, exercised in test_indexed_conv/test_indexed_pool below).
    """
    backend_id = 'Vitis'
    neighbor_indices = _circular_neighbor_map()

    inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_1')(inp)
    kmodel = tf.keras.Model(inputs=inp, outputs=x)

    hls_config = hls4ml.utils.config_from_keras_model(
        kmodel, granularity='name', backend=backend_id, default_precision='ap_fixed<16,6>', default_reuse_factor=1
    )
    hls_config['LayerName']['gather_1']['PackNeighbors'] = True

    output_dir = str(test_root_path / test_case_id)
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel, output_dir=output_dir, backend=backend_id, io_type='io_stream', hls_config=hls_config
    )
    hmodel.write()

    parameters_h = Path(output_dir) / 'firmware' / 'parameters.h'
    params_content = parameters_h.read_text()
    assert 'pack_neighbors  = true' in params_content or 'pack_neighbors = true' in params_content, (
        'pack_neighbors=true not found in generated config -- check '
        "HNeighborGather2DConfigTemplate.format() and the 'PackNeighbors' key resolution."
    )

    defines_h = Path(output_dir) / 'firmware' / 'defines.h'
    myproject_cpp = Path(output_dir) / 'firmware' / 'myproject.cpp'
    call_line = next(line for line in myproject_cpp.read_text().splitlines() if 'neighbor_gather_2d' in line)
    output_type_name = call_line.split(',')[1].strip()

    expected_width = N_NEIGHBORS * N_FEATURES
    defines_content = defines_h.read_text()
    assert f'{output_type_name};' in defines_content, f"Could not find typedef for '{output_type_name}' in {defines_h}."
    assert (
        f'{expected_width}*1> {output_type_name}' in defines_content
        or f'{expected_width}> {output_type_name}' in defines_content
    ), f'Expected {output_type_name} width {expected_width} not found in {defines_h}.'


@pytest.mark.parametrize('pack_neighbors', [False, True])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_indexed_conv(test_case_id, io_type, pack_neighbors):
    """
    IndexedConvolutionLayer, chained after NeighborGatherLayer, both unpacked and packed.
    """
    if pack_neighbors and io_type == 'io_parallel':
        pytest.skip('pack_neighbors is only meaningful for io_stream')

    backend_id = 'Vitis'
    neighbor_indices = _circular_neighbor_map()

    inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_1')(inp)
    x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=N_FILT_1, name='conv_1')(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x = tf.keras.layers.Dense(2, activation='linear', name='output_dense')(x)
    kmodel = tf.keras.Model(inputs=inp, outputs=x)

    hls_config = hls4ml.utils.config_from_keras_model(
        kmodel, granularity='name', backend=backend_id, default_precision='ap_fixed<16,6>', default_reuse_factor=1
    )
    if pack_neighbors:
        hls_config['LayerName']['gather_1']['PackNeighbors'] = True
        hls_config['LayerName']['conv_1']['PackNeighbors'] = True

    output_dir = str(test_root_path / test_case_id)
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel, output_dir=output_dir, backend=backend_id, io_type=io_type, hls_config=hls_config
    )
    hmodel.compile()

    rng = np.random.default_rng(42)
    x_test = rng.standard_normal((4, N_PIXELS, N_FEATURES)).astype(np.float32)
    y_keras = kmodel.predict(x_test, verbose=0)
    y_hls = hmodel.predict(x_test).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=0.1, verbose=True)


@pytest.mark.parametrize('pack_neighbors', [False, True])
@pytest.mark.parametrize('pooling_type', ['max', 'average'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_indexed_pool(test_case_id, io_type, pooling_type, pack_neighbors):
    """
    IndexedPoolingLayer (max and average), chained inside a full
    gather -> conv -> gather -> pool -> gather -> conv pipeline, both
    unpacked and packed.
    """
    if pack_neighbors and io_type == 'io_parallel':
        pytest.skip('pack_neighbors is only meaningful for io_stream')

    backend_id = 'Vitis'
    neighbor_indices = _circular_neighbor_map()

    inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_1')(inp)
    x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=N_FILT_1, name='conv_1')(x)
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_2')(x)
    x = IndexedPoolingLayer(use_3d_conv=False, pooling_type=pooling_type, temporal_pool_size=1, name='pool_1')(x)
    x = NeighborGatherLayer(neighbor_indices=neighbor_indices, use_3d_conv=False, name='gather_3')(x)
    x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=N_FILT_2, name='conv_2')(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x = tf.keras.layers.Dense(2, activation='linear', name='output_dense')(x)
    kmodel = tf.keras.Model(inputs=inp, outputs=x)

    hls_config = hls4ml.utils.config_from_keras_model(
        kmodel, granularity='name', backend=backend_id, default_precision='ap_fixed<20,8>', default_reuse_factor=1
    )
    if pack_neighbors:
        for layer_name in ('gather_1', 'conv_1', 'gather_2', 'pool_1', 'gather_3', 'conv_2'):
            hls_config['LayerName'][layer_name]['PackNeighbors'] = True

    output_dir = str(test_root_path / test_case_id)
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel, output_dir=output_dir, backend=backend_id, io_type=io_type, hls_config=hls_config
    )
    hmodel.compile()

    rng = np.random.default_rng(42)
    x_test = rng.standard_normal((4, N_PIXELS, N_FEATURES)).astype(np.float32)
    y_keras = kmodel.predict(x_test, verbose=0)
    y_hls = hmodel.predict(x_test).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=0.1, verbose=True)


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_full_chain_with_batchnorm(test_case_id):
    """
    Integration test: BatchNormalization -> gather -> conv -> gather -> pool
    -> BatchNormalization -> gather -> conv -> GAP -> Dense, with border
    pixels (-1 neighbor slots) to exercise zero-masking.

    Pass criteria (matching test_minimal.py):
        1. hls4ml conversion and csim compilation succeed without errors.
        2. Agreement between Keras and HLS class predictions (argmax over
           softmax output) >= 95%.
        3. Mean absolute difference in output probabilities < 0.05.

    Precision: 'auto' (hls4ml's built-in InferPrecisionTypes pass) did not
    converge in a reasonable time on this six-layer model and was
    abandoned. Uses a fixed ap_fixed<24,10> instead, rather than a
    separate profiling helper, since a custom precision utility is
    intentionally kept out of the contrib package (see README.md,
    "Choosing precision"). This is an approximation, not a tuned value.
    """
    backend_id = 'Vitis'
    n_test = 20
    neighbor_indices = _circular_neighbor_map(with_border=True)

    def build_model(indices):
        inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
        x = tf.keras.layers.BatchNormalization(name='bn_1')(inp)
        x = NeighborGatherLayer(indices, use_3d_conv=False, name='gather_1')(x)
        x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=N_FILT_1, name='conv_1')(x)
        x = NeighborGatherLayer(indices, use_3d_conv=False, name='gather_2')(x)
        x = IndexedPoolingLayer(use_3d_conv=False, pooling_type='max', temporal_pool_size=1, name='pool_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = NeighborGatherLayer(indices, use_3d_conv=False, name='gather_3')(x)
        x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=N_FILT_2, name='conv_2')(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
        x = tf.keras.layers.Dense(2, activation='linear', name='output')(x)
        return tf.keras.Model(inputs=inp, outputs=x)

    # A single model, built with the real neighbor_indices (-1 included),
    # serves as both the Keras reference and the hls4ml conversion source.
    # NeighborGatherLayer's own zero-masking (a multiplicative mask on
    # index == -1, applied regardless of which "garbage" index tf.gather
    # substitutes internally for -1) already handles border pixels
    # correctly on its own -- no separate "safe indices" substitution is
    # needed or, in fact, correct: substituting -1 -> 0 changes what the
    # layer computes at border pixels (real neighbor-0 data instead of
    # zero), which would make the Keras reference model compute a
    # genuinely different function from the one actually converted to
    # HLS, not just a lower-precision version of it.
    tf.keras.utils.set_random_seed(42)
    kmodel = build_model(neighbor_indices)

    rng = np.random.default_rng(42)
    x_test = rng.standard_normal((n_test, N_PIXELS, N_FEATURES)).astype(np.float32)

    hls_config = hls4ml.utils.config_from_keras_model(
        kmodel, granularity='name', backend=backend_id, default_precision='ap_fixed<24,10>', default_reuse_factor=1
    )
    output_dir = str(test_root_path / test_case_id)
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel, output_dir=output_dir, backend=backend_id, io_type='io_parallel', hls_config=hls_config
    )
    hmodel.compile()

    k_probs = _softmax(kmodel.predict(x_test, verbose=0))
    h_probs = _softmax(hmodel.predict(x_test).reshape(n_test, 2))

    abs_diff = np.abs(k_probs - h_probs)
    agreement = np.mean(np.argmax(k_probs, axis=1) == np.argmax(h_probs, axis=1)) * 100

    assert agreement >= 95.0, f'Classification agreement {agreement:.1f}% is below the 95% threshold.'
    assert abs_diff.mean() < 0.05, f'Mean probability difference {abs_diff.mean():.4f} exceeds the 0.05 threshold.'
