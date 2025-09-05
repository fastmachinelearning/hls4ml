import typing
from collections.abc import Sequence

import numpy as np
from quantizers import get_fixed_quantizer_np

from hls4ml.model.types import FixedPrecisionType

from ._base import KerasV3LayerHandler, QLayerHandler, register

if typing.TYPE_CHECKING:
    import hgq
    from keras import KerasTensor

from decimal import Decimal


@register
class QUnaryLUTHandler(QLayerHandler, KerasV3LayerHandler):
    handles = ('hgq.layers.activation.QUnaryFunctionLUT',)

    def handle(
        self,
        layer: 'hgq.layers.QUnaryFunctionLUT',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from hgq.quantizer.internal import FixedPointQuantizerBase, FloatPointQuantizer
        from keras import ops

        if not layer.enable_iq and not layer.enable_oq:
            raise ValueError('Currently only support input_quantizer enabled UnaryFunctionLUT layer')
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported in QUnaryFunctionLUT layer'

        iq = layer.iq.quantizer
        if isinstance(iq, FixedPointQuantizerBase):
            k, i, f = iq.kif
            mask = k + i + f > 0
            i, f = np.where(mask, i, -32), np.where(mask, f, -32)  # type: ignore
            k, i, f = (Decimal(int(ops.max(x))) for x in (k, i, f))  # type: ignore
            _min = -k * 2**i
            _eps = 2**-f
            _max = 2**i - _eps
            N = (_max - _min) / _eps + 1
            assert float(N).is_integer(), 'Invalid quantizer range'
            N = int(N)
            assert N <= 1e6, 'Too large quantizer range'
            assert np.log2(N).is_integer(), f'Invalid quantizer range: N must be power of 2, got {N}'

            all_inputs = np.linspace(float(_min), float(_max), N, dtype=np.float32)

            config = {}
            config.update(self.default_config)
            table = layer.activation(all_inputs)
            if layer.enable_oq:
                table = layer.oq(table[None, ...])[0]
            table = ops.convert_to_numpy(table)
            if k:
                # idx by binary repr, move the positive part to the front
                table_pos, table_neg = table[N // 2 :], table[: N // 2]
                table = np.concatenate([table_pos, table_neg])
        else:
            raise NotImplementedError('FloatPointQuantizer is not supported yet')

        oq = layer.oq.quantizer
        if isinstance(oq, FixedPointQuantizerBase):
            round_mode = oq.round_mode
            if round_mode.startswith('S_'):
                round_mode = round_mode[2:]
            overflow_mode = oq.overflow_mode
            fixed_q = get_fixed_quantizer_np(round_mode, overflow_mode)
            k, i, f = (ops.convert_to_numpy(x).ravel().item() for x in oq.kif)
            table = fixed_q(table, k, i, f)  # type: ignore

            k, b, I = bool(k), k + i + f, k + i  # noqa: E741
            table_t = FixedPrecisionType(b, I, k)
        else:
            assert isinstance(oq, FloatPointQuantizer)
            raise NotImplementedError('FloatPointQuantizer is not supported yet')

        table = ops.convert_to_numpy(table)

        config.update(
            {
                'class_name': 'UnaryLUT',
                'table_data': table,
                'table_t': table_t,
                'activation': 'unary_lut',
            }
        )

        return (config,)
