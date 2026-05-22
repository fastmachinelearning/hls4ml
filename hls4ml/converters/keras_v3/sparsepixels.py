import math
import typing
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._base import KerasV3LayerHandler

if typing.TYPE_CHECKING:
    import keras
    from keras import KerasTensor

_sparse_context: dict[str, Any] = {}


def _mark_sparse_output(tensor_name: str, n_sparse: int, n_chan: int, height: int, width: int):
    """Record a tensor as coming from a sparse layer so Flatten can be converted."""
    sparse_outputs = _sparse_context.setdefault('sparse_output_tensors', {})
    sparse_outputs[tensor_name] = {
        'n_sparse': n_sparse,
        'n_chan': n_chan,
        'out_height': height,
        'out_width': width,
    }


def _extract_sparse_iq_config(conv_layer, in_tensor_name: str, n_sparse: int, n_chan: int) -> dict[str, Any]:
    """Extract input quantizer config from QConv2D, adapted for sparse tensor shape."""
    from keras import ops

    internal_q = conv_layer._iq.quantizer
    kif_k, kif_i, kif_f = internal_q.kif
    kif_k = np.ravel(ops.convert_to_numpy(kif_k)).astype(np.int16)
    kif_i = np.ravel(ops.convert_to_numpy(kif_i)).astype(np.int16)
    kif_f = np.ravel(ops.convert_to_numpy(kif_f)).astype(np.int16)

    # HGQ quantizers may be per-element (H*W*C); reduce to per-channel
    # Take max of each component independently to get the envelope type
    if kif_k.size > n_chan:
        kif_k = np.max(kif_k.reshape(-1, n_chan), axis=0)
        kif_i = np.max(kif_i.reshape(-1, n_chan), axis=0)
        kif_f = np.max(kif_f.reshape(-1, n_chan), axis=0)

    # Reconstruct KBI from KIF: B = k + i + f, I_bits = k + i
    k = kif_k
    B = kif_k + kif_i + kif_f
    I_bits = kif_k + kif_i

    if k.size > 1:
        k = np.tile(k, n_sparse).reshape(1, -1)
        B = np.tile(B, n_sparse).reshape(1, -1)
        I_bits = np.tile(I_bits, n_sparse).reshape(1, -1)

    overflow_mode: str = internal_q.overflow_mode
    round_mode: str = internal_q.round_mode
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]

    return {
        'name': conv_layer._iq.name,
        'class_name': 'FixedPointQuantizer',
        'mask_kbi': (k, B, I_bits),
        'SAT': overflow_mode,
        'RND': round_mode,
        'fusible': None,
        'input_keras_tensor_names': [in_tensor_name],
        'output_keras_tensor_names': [f'{in_tensor_name}_q'],
        'overrides': {},
    }


def post_process_sparse_layer_list(layer_list: list[dict[str, Any]]) -> None:
    """Convert Reshape (from Flatten) nodes that follow sparse layers into SparseFlatten.
    Called from keras_v3_to_hls after parsing."""
    sparse_outputs = _sparse_context.get('sparse_output_tensors', {})
    if not sparse_outputs:
        return

    for conf in layer_list:
        if conf.get('class_name') != 'Reshape':
            continue
        in_tensors = conf.get('input_keras_tensor_names', [])
        if not in_tensors:
            continue
        src_tensor = in_tensors[0]
        if src_tensor not in sparse_outputs:
            continue
        info = sparse_outputs[src_tensor]
        conf['class_name'] = 'SparseFlatten'
        conf['n_sparse'] = info['n_sparse']
        conf['n_chan'] = info['n_chan']
        conf['out_height'] = info['out_height']
        conf['out_width'] = info['out_width']
        conf.pop('target_shape', None)


class InputReduceHandler(KerasV3LayerHandler):
    handles = ('sparsepixels.layers.InputReduce',)

    def handle(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        in_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        in_height, in_width, n_chan = in_shape

        n_sparse = layer.n_max_pixels
        threshold = float(layer.threshold) if layer.threshold is not None else 0.0

        # Clear any stale state from a previous conversion in the same Python process
        _sparse_context.clear()
        _sparse_context['n_sparse'] = n_sparse
        _sparse_context['spatial'] = (int(in_height), int(in_width))

        for t in out_tensors:
            _mark_sparse_output(t.name, n_sparse, int(n_chan), int(in_height), int(in_width))

        # Hash stores 1-based H and W coordinates separately (see nnet_sparsepixels.h::sparse_input_reduce).
        # Spatial dims only shrink through the network (pooling), so input H/W bound the required bits.
        max_dim = max(int(in_height), int(in_width))
        hash_bits = max(1, math.ceil(math.log2(max_dim + 1)))

        return {
            'class_name': 'SparseInputReduce',
            'in_height': int(in_height),
            'in_width': int(in_width),
            'n_chan': int(n_chan),
            'n_sparse': n_sparse,
            'threshold': threshold,
            'hash_bits': hash_bits,
        }


class QConv2DSparseHandler(KerasV3LayerHandler):
    handles = ('sparsepixels.layers.QConv2DSparse',)

    def handle(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        import keras
        from keras import ops

        conv = layer.conv
        n_chan = int(conv.kernel.shape[2])
        n_filt = int(conv.filters)
        kernel_size = int(conv.kernel_size[0])
        n_sparse = _sparse_context.get('n_sparse', 0)

        if hasattr(conv, 'qkernel'):
            weight_data = ops.convert_to_numpy(conv.qkernel)
        else:
            weight_data = ops.convert_to_numpy(conv.kernel)

        bias_data = None
        if layer._use_bias and hasattr(layer, 'sparse_bias'):
            if hasattr(layer, '_bq'):
                bias_data = ops.convert_to_numpy(layer._bq(layer.sparse_bias))
            else:
                bias_data = ops.convert_to_numpy(layer.sparse_bias)

        name = layer.name
        in_tensor_names = [t.name for t in in_tensors]
        out_tensor_names = [t.name for t in out_tensors]

        iq_conf = None
        has_iq = hasattr(conv, '_iq') and hasattr(conv, '_enable_iq') and conv._enable_iq
        if has_iq:
            iq_conf = _extract_sparse_iq_config(conv, in_tensors[0].name, n_sparse, n_chan)
            in_tensor_names = [f'{in_tensors[0].name}_q']

        config: dict[str, Any] = {
            'class_name': 'SparseConv2D',
            'name': name,
            'n_sparse': n_sparse,
            'n_chan': n_chan,
            'n_filt': n_filt,
            'kernel_size': kernel_size,
            'weight_data': weight_data,
            'bias_data': bias_data,
            'input_keras_tensor_names': in_tensor_names,
            'output_keras_tensor_names': out_tensor_names,
        }

        activation = layer._activation
        spatial = _sparse_context.get('spatial', (1, 1))
        results: list[dict[str, Any]] = []
        if iq_conf is not None:
            results.append(iq_conf)

        if activation not in (None, keras.activations.linear):
            act_name = activation.__name__
            intermediate = f'{out_tensors[0].name}_sparse_act'

            config['output_keras_tensor_names'] = [intermediate]

            act_config: dict[str, Any] = {
                'class_name': 'SparseActivation',
                'name': f'{name}_{act_name}',
                'activation': act_name,
                'n_sparse': n_sparse,
                'n_chan': n_filt,
                'input_keras_tensor_names': [intermediate],
                'output_keras_tensor_names': out_tensor_names,
            }
            for t_name in out_tensor_names:
                _mark_sparse_output(t_name, n_sparse, n_filt, spatial[0], spatial[1])
            results.extend([config, act_config])
            return tuple(results)

        for t_name in out_tensor_names:
            _mark_sparse_output(t_name, n_sparse, n_filt, spatial[0], spatial[1])
        results.append(config)
        return tuple(results)


class AveragePooling2DSparseHandler(KerasV3LayerHandler):
    handles = ('sparsepixels.layers.AveragePooling2DSparse',)

    def handle(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        pool_size = int(layer.avg_pool.pool_size[0])

        feat_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        n_chan = int(feat_shape[-1])
        n_sparse = _sparse_context.get('n_sparse', 0)

        prev_h, prev_w = _sparse_context.get('spatial', (1, 1))
        new_h, new_w = prev_h // pool_size, prev_w // pool_size
        _sparse_context['spatial'] = (new_h, new_w)

        out_tensor_names = [t.name for t in out_tensors]
        for t_name in out_tensor_names:
            _mark_sparse_output(t_name, n_sparse, n_chan, new_h, new_w)

        return {
            'class_name': 'SparsePooling2D',
            'n_sparse': n_sparse,
            'n_chan': n_chan,
            'pool_size': pool_size,
        }
