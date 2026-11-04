"""
run_io_stream.py (xcku040)
============================
End-to-end conversion, validation and (optional) synthesis of a small,
single-conv CTLearn-style model, targeting the Xilinx XCKU040.

Model architecture
-------------------
    Input(163, 20)
    -> BatchNormalization
    -> NeighborGatherLayer
    -> IndexedConvolutionLayer(8)
    -> BatchNormalization
    -> GlobalAveragePooling1D
    -> Dense(64, relu)
    -> Dense(32, relu)
    -> Dense(2, linear)

The neighbor map and input data are both generated synthetically (see
"Synthetic data" below), and the model uses randomly-initialized
(untrained) weights, so this example is self-contained: it needs no
external model or data files, and the pipeline it exercises works with
any neighbor-indexing scheme of this shape, not just a specific
camera's real geometry.

The Softmax layer is excluded from the hls4ml model and applied in
Python after inference, to avoid io_parallel/io_stream Softmax
compatibility issues (see the package README, "Known limitations").

This is a separate, smaller model from the one in `../ctlearn/`, with
a single conv block and no pooling, sized specifically to fit real C
synthesis on this target.

Synthetic data
--------------
The neighbor map (163 pixels, 7 neighbors) is generated procedurally
by a circular-shift pattern, with the last 90 neighbor slots (across
the last several pixels) set to -1 to exercise border-pixel zero-
masking, matching the border-slot count of the real camera geometry
this backbone was originally developed against. Input data is drawn
from a fixed-seed standard normal distribution. Since the model is
untrained, its output has no physical meaning; only the numerical
agreement between the Keras and HLS computations is being validated
here, not classification accuracy.

Usage
-----
    python run_io_stream.py                    # conversion + csim only
    python run_io_stream.py --synth             # + C synthesis (hmodel.build())
    python run_io_stream.py --fifo-opt           # + cosim-based FIFO depth optimization (implies --synth)

Requires Vitis HLS 2025.2 and Vivado 2023.2 active in PATH for
--synth to work correctly.

Pass criteria (steps [1/5]-[5/5])
------------------------------------
    - Mean absolute difference in softmax probabilities < 0.05
    - Classification agreement between Keras and HLS csim >= 95 %
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description='io_stream integration test for the xcku040 single-conv model')
parser.add_argument(
    '--synth',
    action='store_true',
    help='Also run C synthesis (hmodel.build()) after csim validation. '
    'Requires Vitis HLS 2025.2 and Vivado 2023.2 active in PATH.',
)
parser.add_argument(
    '--rf',
    type=int,
    default=1,
    help='Global reuse_factor passed to get_hls_config() (default: 1). '
    'Valid values for IndexedConvolutionLayer (io_stream) are '
    'constrained by nnet::dense_resource: must satisfy '
    '(n_in*n_out) %% multfactor == 0, or be a divisor/multiple of '
    'n_features_in (20 in this model) -- e.g. 1, 2, 4, 5, 10, 20.',
)
parser.add_argument(
    '--layer-rf',
    action='append',
    default=[],
    metavar='LAYER:RF[:STRATEGY]',
    help='Per-layer ReuseFactor override, e.g. --layer-rf fc_type_2:8:Resource. '
    'Strategy defaults to Latency if omitted. Repeatable.',
)
parser.add_argument(
    '--pack-neighbors',
    action='store_true',
    dest='pack_neighbors',
    help='Enable pack_neighbors=True on neighbor_gather_layer and its '
    'direct consumer (SingleCNNIndexed_block_conv_1_1), for a '
    'single wide stream packet per pixel instead of one packet '
    'per neighbor. Reduces gather latency from ~n_pixels*n_neighbors '
    'to ~n_pixels cycles.',
)
parser.add_argument(
    '--fifo-opt',
    action='store_true',
    dest='fifo_opt',
    help='Run cosim-based FIFO depth optimization (vitis:fifo_depth_optimization '
    'flow) during conversion, using real input data as the testbench '
    'stimulus. Implies --synth (a final synthesis with the corrected FIFO '
    'depths is still required to get an accurate resource report). '
    'Significantly slower than --synth alone, since it runs its own '
    'synth+cosim pass internally in addition to the final one.',
)
args = parser.parse_args()
if args.fifo_opt:
    args.synth = True

# examples/ (one directory up) holds precision_utils.py, shared by both examples
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from precision_utils import get_hls_config  # noqa: E402

import hls4ml  # noqa: E402
from hls4ml.contrib.indexed_geometry import IndexedConvolutionLayer, NeighborGatherLayer  # noqa: E402

# Paths (relative to this file)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, 'hls4mlprj_xcku040_stream')

N_PIXELS = 163
N_NEIGHBORS = 7
N_FEATURES = 20
N_BORDER_SLOTS = 90
N_TEST = 30

PASS = '\033[92m\u2713  PASSED\033[0m'
FAIL = '\033[91m\u2717  FAILED\033[0m'

print(f'\n  Using reuse_factor={args.rf}' + (' (synthesis enabled)' if args.synth else ''))

# [1/5] Neighbor map
print('=' * 65)
print('  [1/5] Generating synthetic neighbor map...')
print('=' * 65)

# Circular-shift pattern: pixel p's k-th neighbor is (p + k + 1) % N_PIXELS,
# so every pixel has N_NEIGHBORS valid neighbors by construction. The last
# N_BORDER_SLOTS slots (across the last few pixels) are set to -1, to
# exercise the same border-pixel zero-masking path a real, irregularly-
# shaped camera geometry would need.
neighbor_indices = np.stack([np.roll(np.arange(N_PIXELS), -k) for k in range(1, N_NEIGHBORS + 1)], axis=1).astype(np.int32)
flat_view = neighbor_indices.reshape(-1)
flat_view[-N_BORDER_SLOTS:] = -1

print(f'  Shape: {neighbor_indices.shape}')
print(f'  Border slots (-1): {(neighbor_indices == -1).sum()}')

# [2/5] Input data
print('\n' + '=' * 65)
print('  [2/5] Generating synthetic input data...')
print('=' * 65)

rng = np.random.default_rng(42)
x_test = rng.standard_normal((N_TEST, N_PIXELS, N_FEATURES)).astype(np.float32)
print(f'  Shape: {x_test.shape}')
print(f'  Using {N_TEST} events for the test')

# [3/5] Keras model
print('\n' + '=' * 65)
print('  [3/5] Building Keras model (random weights)...')
print('=' * 65)

# Random, untrained weights: only the numerical agreement between the
# Keras and HLS computations is being validated here, not classification
# accuracy. The seed makes the weights (and therefore the results below)
# reproducible across runs.
tf.keras.utils.set_random_seed(42)

# The Softmax layer is omitted here; it is applied in Python after
# inference.
inp = tf.keras.Input(shape=(N_PIXELS, N_FEATURES), name='input_1')
x = tf.keras.layers.BatchNormalization(momentum=0.99, name='batch_normalization')(inp)
x = NeighborGatherLayer(neighbor_indices, use_3d_conv=False, name='neighbor_gather_layer')(x)
x = IndexedConvolutionLayer(use_3d_conv=False, temporal_kernel_size=1, filters=8, name='SingleCNNIndexed_block_conv_1_1')(x)
x = tf.keras.layers.BatchNormalization(momentum=0.99, name='batch_normalization_1')(x)
x = tf.keras.layers.GlobalAveragePooling1D(name='SingleCNNIndexed_block_global_avgpool')(x)
x = tf.keras.layers.Dense(64, activation='relu', name='fc_type_1')(x)
x = tf.keras.layers.Dense(32, activation='relu', name='fc_type_2')(x)
x = tf.keras.layers.Dense(2, activation='linear', name='type')(x)
kmodel = tf.keras.Model(inputs=inp, outputs=x, name='CTLearn_model')

print('  Model built with random weights and without Softmax layer')
kmodel.summary()

# [4/5] hls4ml conversion
print('\n' + '=' * 65)
print('  [4/5] Converting to hls4ml (io_stream) and compiling csim...')
print('=' * 65)

hls_config = get_hls_config(
    kmodel,
    x_sample=x_test[:10],
    reuse_factor=args.rf,
    verbose=True,
)

# Per-layer overrides, applied on top of the global default set above.
for spec in args.layer_rf:
    parts = spec.split(':')
    layer_name, rf = parts[0], int(parts[1])
    strategy = parts[2] if len(parts) > 2 else 'Latency'
    hls_config['LayerName'][layer_name]['ReuseFactor'] = rf
    hls_config['LayerName'][layer_name]['Strategy'] = strategy
    print(f'  Override: {layer_name} -> ReuseFactor={rf}, Strategy={strategy}')

# Packed gather output: NeighborGatherLayer and its direct consumer
# must have matching pack_neighbors settings, or conversion will fail
# with a clear shape-mismatch AssertionError (see
# HIndexedConv2D.initialize() / HIndexedPool2D.initialize()).
if args.pack_neighbors:
    hls_config['LayerName']['neighbor_gather_layer']['PackNeighbors'] = True
    hls_config['LayerName']['SingleCNNIndexed_block_conv_1_1']['PackNeighbors'] = True
    print('  Override: neighbor_gather_layer + SingleCNNIndexed_block_conv_1_1 -> PackNeighbors=True')

if args.fifo_opt:
    # cosim-based FIFO depth optimization needs a real input sample on
    # disk (.npy, at least 2 events) as the C++ testbench stimulus, so
    # profiled depths reflect actual data instead of empty/default
    # input. x_test (synthetic events, generated in [2/5]) isn't accepted
    # directly by hls4ml's writer (only .dat or .npy), so it's re-saved
    # here as .npy.
    tb_input_path = os.path.join(HERE, 'tb_input_features.npy')
    np.save(tb_input_path, x_test)
    hls_config['InputData'] = tb_input_path
    hls_config['Flows'] = ['vitis:fifo_depth_optimization']
    print(f'  FIFO depth optimization enabled (cosim), input data: {tb_input_path}')

hmodel = hls4ml.converters.convert_from_keras_model(
    kmodel,
    output_dir=OUTPUT_DIR,
    backend='Vitis',
    io_type='io_stream',
    hls_config=hls_config,
    part='xcku040-ffva1156-2-i',
    clock_period=3.03,
    clock_uncertainty='0.5ns',
)
hmodel.write()

# Show the generated calls for our custom layers, to confirm the
# io_stream overloads were used correctly
myproject_cpp = os.path.join(OUTPUT_DIR, 'firmware', 'myproject.cpp')
print(f'\n--- Generated {myproject_cpp} (custom-layer excerpt) ---')
with open(myproject_cpp) as f:
    for line in f:
        if any(kw in line for kw in ['neighbor_gather', 'indexed_conv', 'indexed_maxpool', 'indexed_avgpool']):
            print(' ', line.rstrip())

hmodel.compile()
print('  csim compiled successfully')

# [5/5] Prediction comparison
print('\n' + '=' * 65)
print('  [5/5] Comparing predictions...')
print('=' * 65)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# Keras reference predictions (Softmax applied in Python
# for consistency with how hls4ml predictions are computed below)
k_probs = softmax(kmodel.predict(x_test, verbose=0))

# hls4ml predictions -- apply Softmax in Python
h_probs = softmax(hmodel.predict(x_test).reshape(N_TEST, 2))

abs_diff = np.abs(k_probs - h_probs)
k_classes = np.argmax(k_probs, axis=1)
h_classes = np.argmax(h_probs, axis=1)
agreement = np.mean(k_classes == h_classes) * 100

print(f'\n  {"Event":>6}  {"Keras p(1)":>12}  {"HLS p(1)":>12}  {"|diff|":>10}  {"Match":>6}')
print(f'  {"─" * 55}')
for i in range(min(20, N_TEST)):
    match = '✓' if k_classes[i] == h_classes[i] else '✗'
    print(f'  {i:>6}  {k_probs[i, 1]:>12.6f}  {h_probs[i, 1]:>12.6f}  {abs_diff[i].max():>10.6f}  {match:>6}')
if N_TEST > 20:
    print(f'  ... ({N_TEST - 20} more events)')

print('\n  Softmax probability difference:')
print(f'    Max  diff: {abs_diff.max():.6f}')
print(f'    Mean diff: {abs_diff.mean():.6f}')
print(f'\n  Classification agreement: {agreement:.1f}%  ({int(agreement * N_TEST / 100)}/{N_TEST} events)')

ok_diff = abs_diff.mean() < 0.05
ok_agreement = agreement >= 95.0

print(f'\n  {"─" * 55}')
print(f'  Mean diff < 0.05 :  {PASS if ok_diff else FAIL}  ({abs_diff.mean():.4f})')
print(f'  Agreement >= 95% :  {PASS if ok_agreement else FAIL}  ({agreement:.1f}%)')

print('\n' + '=' * 65)
if ok_diff and ok_agreement:
    print(f'  {PASS} -- xcku040 io_stream csim validation completed')
    print(f'  HLS project: {OUTPUT_DIR}')
else:
    print(f'  {FAIL} -- Review metrics above before proceeding')
    raise AssertionError('run_io_stream (xcku040) failed (csim)')
print('=' * 65)

# [6/6] Synthesis (optional, --synth flag)
if args.synth:
    print('\n' + '=' * 65)
    print('  [6/6] Running C synthesis...')
    print('=' * 65)
    print('  No source patches applied at this stage. If csynth fails,')
    print('  inspect the error below before patching anything.')

    hmodel.build(
        reset=True,
        csim=False,
        synth=True,
        cosim=False,
        export=False,
    )
    report = hls4ml.report.read_vivado_report(OUTPUT_DIR)
    print(report)
else:
    print('\n  (Skipping C synthesis. Re-run with --synth to enable it.)')
