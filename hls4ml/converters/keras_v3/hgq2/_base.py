from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from hls4ml.converters.keras_v3._base import KerasV3LayerHandler, register
from hls4ml.converters.keras_v3.conv import ConvHandler
from hls4ml.converters.keras_v3.core import ActivationHandler, DenseHandler
from hls4ml.converters.keras_v3.einsum_dense import EinsumDenseHandler
from hls4ml.converters.keras_v3.merge import MergeHandler

if TYPE_CHECKING:
    import hgq
    from keras import KerasTensor
    from keras.src.layers.layer import Layer as Layer


def extract_fixed_quantizer_config(q, tensor: 'KerasTensor', is_input: bool) -> dict[str, Any]:
    from hgq.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerKBI, FixedPointQuantizerKIF
    from keras import ops

    internal_q: FixedPointQuantizerKIF | FixedPointQuantizerKBI = q.quantizer

    shape: tuple[int, ...] = tensor.shape[1:]  # type: ignore
    if any([s is None for s in shape]):
        raise ValueError(f"Tensor {tensor.name} has at least one dimension with no fixed size")
    k, i, f = internal_q.kif
    k, B, I = k, k + i + f, k + i  # type: ignore # noqa: E741
    k, B, I = ops.convert_to_numpy(k), ops.convert_to_numpy(B), ops.convert_to_numpy(I)  # noqa: E741
    I = np.where(B > 0, I, 0)  # noqa: E741 # type: ignore

    k = np.broadcast_to(k.astype(np.int16), (1,) + shape)  # type: ignore
    B = np.broadcast_to(B.astype(np.int16), (1,) + shape)  # type: ignore
    I = np.broadcast_to(I.astype(np.int16), (1,) + shape)  # noqa: E741

    overflow_mode: str = internal_q.overflow_mode
    round_mode: str = internal_q.round_mode
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]
    fusible = np.unique(k).size == 1 and np.unique(B).size == 1 and np.unique(I).size == 1

    input_keras_tensor_names = tensor.name if is_input else f'{tensor.name}_q'
    output_keras_tensor_names = f'{tensor.name}_q' if is_input else tensor.name
    return {
        'name': q.name,
        'class_name': 'FixedPointQuantizer',
        'mask_kbi': (k, B, I),
        'SAT': overflow_mode,
        'RND': round_mode,
        'fusible': fusible,
        'input_keras_tensor_names': [input_keras_tensor_names],
        'output_keras_tensor_names': [output_keras_tensor_names],
        'overrides': {},
    }


def override_io_tensor_confs(confs: tuple[dict[str, Any], ...], overrides: dict[str, str]):
    for conf in confs:
        inp_tensor_names = conf['input_keras_tensor_names']
        out_tensor_names = conf['output_keras_tensor_names']
        conf['input_keras_tensor_names'] = [overrides.get(name, name) for name in inp_tensor_names]
        conf['output_keras_tensor_names'] = [overrides.get(name, name) for name in out_tensor_names]


@register
class QLayerHandler(KerasV3LayerHandler):
    def __call__(
        self,
        layer: 'hgq.layers.QLayerBase',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        ret = super().__call__(layer, in_tensors, out_tensors)

        if layer._enable_iq and hasattr(layer, '_iq'):
            if len(in_tensors) > 1:
                iq_confs = [extract_fixed_quantizer_config(q, tensor, True) for q, tensor in zip(layer._iq, in_tensors)]
            else:
                iq_confs = [extract_fixed_quantizer_config(layer._iq, in_tensors[0], True)]
        else:
            iq_confs = ()

        if layer._enable_oq:
            if len(out_tensors) > 1:
                oq_confs = [extract_fixed_quantizer_config(q, tensor, False) for q, tensor in zip(layer._oq, out_tensors)]
            else:
                oq_confs = [extract_fixed_quantizer_config(layer._oq, out_tensors[0], False)]
        else:
            oq_confs = ()

        if iq_confs:
            _froms = [t.name for t in in_tensors]
            _tos = [f'{t.name}_q' for t in in_tensors]
            overrides = dict(zip(_froms, _tos))
            override_io_tensor_confs(ret, overrides)

        if oq_confs:
            _froms = [t.name for t in out_tensors]
            _tos = [f'{t.name}_q' for t in out_tensors]
            overrides = dict(zip(_froms, _tos))
            override_io_tensor_confs(ret, overrides)

        return *iq_confs, *ret, *oq_confs

    def load_weight(self, layer: 'Layer', key: str):
        from keras import ops

        if hasattr(layer, f'q{key}'):
            return ops.convert_to_numpy(getattr(layer, f'q{key}'))
        return super().load_weight(layer, key)

    def default_class_name(self, layer: 'Layer') -> str:
        class_name = layer.__class__.__name__
        if class_name.startswith('Q'):
            class_name = class_name[1:]
        return class_name


@register
class QEinsumDenseHandler(QLayerHandler, EinsumDenseHandler):
    handles = (
        'hgq.layers.core.einsum_dense.QEinsumDense',
        'hgq.layers.einsum_dense_batchnorm.QEinsumDenseBatchnorm',
    )


@register
class QStandaloneQuantizerHandler(KerasV3LayerHandler):
    handles = ('hgq.quantizer.quantizer.Quantizer',)

    def handle(
        self,
        layer: 'hgq.quantizer.Quantizer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = extract_fixed_quantizer_config(layer, in_tensors[0], True)
        del conf['output_keras_tensor_names']
        return conf


@register
class QConvHandler(QLayerHandler, ConvHandler):
    handles = (
        'hgq.layers.conv.QConv1D',
        'hgq.layers.conv.QConv2D',
        # 'hgq.layers.conv.QConv3D',
    )

    def handle(
        self,
        layer: 'hgq.layers.QConv1D|hgq.layers.QConv2D',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = super().handle(layer, in_tensors, out_tensors)
        pf = layer.parallelization_factor
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore
        if pf < 0:
            if layer.data_format == 'channels_last':
                pf = prod(out_shape[:-1])
            else:
                pf = prod(out_shape[1:])
        conf['parallelization_factor'] = pf
        return conf


@register
class QDenseHandler(QLayerHandler, DenseHandler):
    handles = ('hgq.layers.core.dense.QDense', 'hgq.layers.core.dense.QBatchNormDense')

    def handle(
        self,
        layer: 'hgq.layers.QDense',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = super().handle(layer, in_tensors, out_tensors)
        conf['class_name'] = 'Dense'
        in_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        if len(in_shape) > 1:
            pf = layer.parallelization_factor
            conf['parallelization_factor'] = pf
        return conf


@register
class QActivationHandler(QLayerHandler, ActivationHandler):
    handles = ('hgq.layers.activation.QActivation',)


@register
class QBatchNormalizationHandler(QLayerHandler):
    handles = ('hgq.layers.batch_normalization.QBatchNormalization',)

    def handle(
        self,
        layer: 'hgq.layers.QBatchNormalization',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from keras import ops

        scale, offset = layer.qscaler_and_qoffset
        scale = ops.convert_to_numpy(scale)
        offset = ops.convert_to_numpy(offset)

        assert layer.axis in (len(in_tensors[0].shape) - 1, -1), 'Only batch_norm with axis=-1 is supported in hls4ml'

        return {
            'n_filt': scale.size,  # type: ignore
            'n_in': prod(in_tensors[0].shape[1:]),  # type: ignore
            'scale_data': scale,
            'bias_data': offset,
        }


@register
class QMergeHandler(QLayerHandler, MergeHandler):
    handles = (
        'hgq.layers.ops.merge.QAdd',
        'hgq.layers.ops.merge.QSubtract',
        'hgq.layers.ops.merge.QMultiply',
        'hgq.layers.ops.merge.QAverage',
        'hgq.layers.ops.merge.QMaximum',
        'hgq.layers.ops.merge.QMinimum',
        'hgq.layers.ops.merge.QAveragePow2',
        'hgq.layers.ops.merge.QDot',
    )

    def handle(
        self,
        layer: 'hgq.layers.ops.merge.QMerge',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        cls_name = layer.__class__.__name__[1:]
        conf = super().handle(layer, in_tensors, out_tensors, cls_name)
        if cls_name == 'AveragePow2':
            conf['op'] = 'average'
            conf['scale'] = layer._scale
        elif cls_name == 'Dot':
            msg = (
                f'Dot operation in hls4ml only supports two flatten, identical tensors. Got '
                f'{in_tensors[0].shape} and {in_tensors[1].shape} for layer {layer.name}.'
            )
            assert all(len(t.shape) == 2 for t in in_tensors) and in_tensors[0].shape == in_tensors[1].shape, msg
            conf['class_name'] = 'Dot'
            conf['op'] = 'dot1d'
            conf['axes'] = layer.axes
        return conf
