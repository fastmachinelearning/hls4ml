from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np

from hls4ml.converters.keras_v3._base import KerasV3LayerHandler, register
from hls4ml.converters.keras_v3.conv import ConvHandler
from hls4ml.converters.keras_v3.core import ActivationHandler, DenseHandler
from hls4ml.converters.keras_v3.hgq2._base import override_io_tensor_confs

if TYPE_CHECKING:
    import pquant
    from keras import KerasTensor
    from keras.src.layers.layer import Layer as Layer


def extract_quantizer_config(
    q, extract_kif, tensor: 'KerasTensor', is_input: bool, overflow_attr: str = 'overflow_mode'
) -> dict[str, Any]:
    from keras import ops

    shape: tuple[int, ...] = tensor.shape[1:]  # type: ignore
    if any([s is None for s in shape]):
        raise ValueError(f'Tensor {tensor.name} has at least one dimension with no fixed size')

    k, i, f = extract_kif(q)
    k, B, I = k, k + i + f, k + i  # type: ignore # noqa: E741
    k, B, I = ops.convert_to_numpy(k), ops.convert_to_numpy(B), ops.convert_to_numpy(I)  # noqa: E741
    I = np.where(B > 0, I, 0)  # noqa: E741 # type: ignore

    k = np.broadcast_to(k.astype(np.int16), (1,) + shape)  # type: ignore
    B = np.broadcast_to(B.astype(np.int16), (1,) + shape)  # type: ignore
    I = np.broadcast_to(I.astype(np.int16), (1,) + shape)  # noqa: E741

    overflow_mode: str = getattr(q, overflow_attr, 'SAT')
    round_mode: str = q.round_mode
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


def extract_pquant_quantizer_config(q, tensor: 'KerasTensor', is_input: bool) -> dict[str, Any]:
    from pquant.quantizer import Quantizer

    if not isinstance(q, Quantizer):
        raise TypeError(f'Quantizer {type(q).__name__} ({q.__module__}) is not an instance of any allowed Quantizer class.')

    if q.use_hgq:
        return extract_quantizer_config(q.quantizer.quantizer, lambda q: q.kif, tensor, is_input)
    else:
        return extract_quantizer_config(q, lambda q: (q.k, q.i, q.f), tensor, is_input, 'overflow')


@register
class PQLayerHandler(KerasV3LayerHandler):
    def __call__(
        self,
        layer: (
            'pquant.core.keras.layers.PQWeightBiasBase | '
            'pquant.core.keras.layers.PQBatchNormalization | '
            'pquant.core.keras.layers.QuantizedPooling | '
            'pquant.core.keras.layers.QuantizedActivation'
        ),
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        ret = super().__call__(layer, in_tensors, out_tensors)

        if getattr(layer, 'quantize_input', False) and hasattr(layer, 'input_quantizer'):
            if len(in_tensors) > 1:
                iq_confs = [
                    extract_pquant_quantizer_config(q, tensor, True) for q, tensor in zip(layer.input_quantizer, in_tensors)
                ]
            else:
                iq_confs = [extract_pquant_quantizer_config(layer.input_quantizer, in_tensors[0], True)]
        else:
            iq_confs = ()

        if getattr(layer, 'quantize_output', False) and hasattr(layer, 'output_quantizer'):
            if len(out_tensors) > 1:
                oq_confs = [
                    extract_pquant_quantizer_config(q, tensor, False)
                    for q, tensor in zip(layer.output_quantizer, out_tensors)
                ]
            else:
                oq_confs = [extract_pquant_quantizer_config(layer.output_quantizer, out_tensors[0], False)]
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
        if class_name.startswith('PQ'):
            class_name = class_name[2:]
        return class_name


@register
class PQActivationHandler(PQLayerHandler, ActivationHandler):
    handles = ('pquant.core.keras.activations.PQActivation',)

    def handle(
        self,
        layer: 'pquant.core.keras.activations.PQActivation',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        activation = getattr(layer, 'activation_name', 'linear')
        match activation:
            case 'hard_tanh':
                class_name = 'HardActivation'
            case _:
                class_name = 'Activation'

        config['activation'] = activation
        config['class_name'] = class_name
        config['n_in'] = prod(in_tensors[0].shape[1:])  # type: ignore
        return (config,)


@register
class PQBatchNormalizationHandler(PQLayerHandler):
    handles = ('pquant.core.keras.layers.PQBatchNormalization',)

    def handle(
        self,
        layer: 'pquant.core.keras.layers.PQBatchNormalization',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from keras import ops

        assert layer.axis in (len(in_tensors[0].shape) - 1, -1), 'Only batch_norm with axis=-1 is supported in hls4ml'

        conf = {}
        conf['class_name'] = layer.__class__.__name__[1:]
        conf['n_in'] = prod(in_tensors[0].shape[1:])

        conf['use_gamma'] = layer.scale
        if conf['use_gamma']:
            conf['gamma_data'] = ops.convert_to_numpy(layer.weight_quantizer(layer.gamma))
        else:
            conf['gamma_data'] = 1

        conf['use_beta'] = layer.center
        if conf['use_beta']:
            conf['beta_data'] = ops.convert_to_numpy(layer.bias_quantizer(layer.beta))
        else:
            conf['beta_data'] = 0

        conf['mean_data'] = ops.convert_to_numpy(layer.moving_mean)
        conf['variance_data'] = ops.convert_to_numpy(layer.moving_variance)
        conf['n_filt'] = conf['variance_data'].size

        return conf


@register
class PQConvHandler(PQLayerHandler, ConvHandler):
    handles = ('pquant.core.keras.layers.PQConv1d', 'pquant.core.keras.layers.PQConv2d')

    def handle(
        self,
        layer: 'pquant.core.keras.layers.PQConv1D | pquant.core.keras.layers.PQConv2D',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = super().handle(layer, in_tensors, out_tensors)
        conf['class_name'] = layer.__class__.__name__[1:-1] + 'D'
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
class PQDenseHandler(PQLayerHandler, DenseHandler):
    handles = ('pquant.core.keras.layers.PQDense',)

    def handle(
        self,
        layer: 'pquant.core.keras.layers.PQDense',
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
