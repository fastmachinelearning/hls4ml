import typing
from collections.abc import Sequence
from math import prod

from hls4ml.model.types import FixedPrecisionType, RoundingMode, SaturationMode

from ._base import QLayerHandler, register

if typing.TYPE_CHECKING:
    import hgq
    from hgq.quantizer.internal import FixedPointQuantizerBase
    from keras import KerasTensor


def fixed_quantizer_to_hls4ml_t(q: 'FixedPointQuantizerBase', take_max=False):
    from keras import ops

    k, i, f = q.kif
    k = ops.convert_to_numpy(k)
    i = ops.convert_to_numpy(i)
    f = ops.convert_to_numpy(f)
    if not take_max:
        assert k.size == 1 and i.size == 1 and f.size == 1, 'Only homogeneous quantizer is supported'
        k = bool(k.ravel().item())
        i = int(i.ravel().item())
        f = int(f.ravel().item())
    else:
        k = bool(k.max())
        i = int(i.max())
        f = int(f.max())

    k, b, I = k, k + i + f, k + i  # noqa: E741
    b = max(1, b)
    round_mode = q.round_mode
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]  # stochastic rounding
    round_mode = getattr(RoundingMode, round_mode)
    sat_mode = getattr(SaturationMode, q.overflow_mode)
    return FixedPrecisionType(b, I, k, rounding_mode=round_mode, saturation_mode=sat_mode)


@register
class QSoftmaxHandler(QLayerHandler):
    handles = ('hgq.layers.softmax.QSoftmax',)

    def handle(
        self,
        layer: 'hgq.layers.QSoftmax',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported in QSoftmax layer'
        if len(layer.axes) == 1:
            ax = layer.axes[0]
            ax = ax if ax >= 0 else len(in_tensors[0].shape) + ax
            # io_stream asserts axes=-1, convert to -1 when it is
            n_outer: int = prod(in_tensors[0].shape[1:ax])  # type: ignore
            n_inner: int = prod(in_tensors[0].shape[ax + 1 :])  # type: ignore
            ax = -1 if ax == len(in_tensors[0].shape) - 1 else ax
        else:  # softmax along multiple axes
            axs = [ax if ax >= 0 else len(in_tensors[0].shape) + ax for ax in layer.axes]
            axs = sorted(axs)
            assert all(ax1 - ax0 == 1 for ax0, ax1 in zip(axs[:-1], axs[1:])), 'Softmax must act on adjacent axes'
            n_outer: int = prod(in_tensors[0].shape[1 : axs[0]])  # type: ignore
            n_inner: int = prod(in_tensors[0].shape[axs[-1] + 1 :])  # type: ignore
            ax = -1
        n_in: int = prod(in_tensors[0].shape[1:])  # type: ignore

        from hgq.quantizer.internal import FixedPointQuantizerBase
        from keras import ops

        impl = 'stable' if layer.stable else 'latency'

        if impl == 'stable':
            exp_table_size = 2 ** int(ops.convert_to_numpy(ops.max(layer.exp_table.iq.quantizer.bits)))  # type: ignore
        else:
            exp_table_size = None  # Placeholder, will be overridden in bit-exact pass

        exp_oq = layer.exp_table.oq.quantizer
        inv_oq = layer.inv_table.oq.quantizer
        inv_iq = layer.inv_table.iq.quantizer
        assert isinstance(exp_oq, FixedPointQuantizerBase), 'Only fixed-point quantizer is supported for exp_table'
        exp_table_t = fixed_quantizer_to_hls4ml_t(exp_oq)
        inv_table_t = fixed_quantizer_to_hls4ml_t(inv_oq)
        inv_inp_t = fixed_quantizer_to_hls4ml_t(inv_iq)
        exp_scale = layer.input_scaler

        inv_table_size = 2**inv_inp_t.width

        parallelization_factor = layer.parallelization_factor

        if parallelization_factor < 0:
            parallelization_factor = n_outer * n_inner

        if len(in_tensors) == 2:
            raise NotImplementedError("Masked softmax not supported yet")
            class_name = 'MaskedSoftmax'
        elif len(in_tensors) == 1:
            class_name = 'Softmax'
        else:
            raise ValueError(f"Too many inputs for softmax layer {layer.name}: expected 1 or 2, got {len(in_tensors)}")

        config = {}
        config.update(self.default_config)
        config.update(
            {
                'axis': ax,
                'n_in': n_in,
                'activation': 'softmax',
                'n_outer': n_outer,
                'n_inner': n_inner,
                'implementation': impl,
                'exp_table_t': exp_table_t,
                'exp_table_size': exp_table_size,
                'inv_table_t': inv_table_t,
                'inv_table_size': inv_table_size,
                'inv_inp_t': inv_inp_t,
                'exp_scale': exp_scale,
                'parallelization_factor': parallelization_factor,
                'class_name': class_name,
                '_bit_exact': True,
            }
        )

        if layer.stable:
            inp_norm_t = fixed_quantizer_to_hls4ml_t(layer.exp_table.iq.quantizer)
            config['inp_norm_t'] = inp_norm_t

        return (config,)
