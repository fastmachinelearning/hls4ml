"""
This pass compares each emitted softmax type against the HGQ quantizer stashed by
hls4ml.converters.keras_v3.hgq2.softmax and reports any ceiling or grid difference, plus any
LUT that is too small to address its own input type.
"""

from warnings import warn

from hls4ml.model.layers import Softmax
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, NamedType


def _emitted_kif(t):
    """(k, i, f) actually emitted, in HGQ convention (i excludes the sign bit)."""
    p: FixedPrecisionType = t.precision if isinstance(t, NamedType) else t
    k = int(bool(p.signed))
    return k, p.integer - k, p.width - p.integer


class ValidateHgqSoftmaxTypes(OptimizerPass):
    def match(self, node):
        return isinstance(node, Softmax) and bool(node.get_attr('_hgq_ref_kif'))

    def transform(self, model, node):
        ref = node.get_attr('_hgq_ref_kif')
        problems = []

        for name, (_rk, ri, rf) in ref.items():
            t = node.attributes.get(name)
            if t is None:
                continue
            _ek, ei, ef = _emitted_kif(t)
            # The sign bit may legitimately be dropped for a non-negative quantity; the
            # ceiling (2**i) and the grid (2**-f) are what have to agree.
            if ei != ri:
                problems.append(
                    f'{name}: saturates at 2**{ei} but HGQ saturates at 2**{ri} '
                    f'-- HLS and Keras will disagree on every value that clips'
                )
            if ef != rf:
                problems.append(f'{name}: grid is 2**-{ef} but HGQ quantizes to 2**-{rf}')

        # A LUT must be able to address every distinct value of its input type, otherwise
        # softmax_idx_from_real_val() drops the low index bits.
        for size_attr, inp_attr in (('exp_table_size', 'inp_norm_t'), ('inv_table_size', 'inv_inp_t')):
            size = node.get_attr(size_attr)
            inp_t = node.attributes.get(inp_attr)
            if size is None or inp_t is None:
                continue
            needed = 2 ** int(inp_t.precision.width)
            if int(size) < needed:
                dropped = (needed // max(int(size), 1)).bit_length() - 1
                problems.append(
                    f'{size_attr}={size} cannot address {inp_attr} '
                    f'(ac_fixed<{inp_t.precision.width},{inp_t.precision.integer}>, needs {needed}): '
                    f'the low {dropped} index bit(s) are discarded, coarsening the lookup'
                )

        if problems:
            warn(
                f'Softmax layer {node.name} will not be bit-exact with the Keras model:\n  - ' + '\n  - '.join(problems),
                stacklevel=1,
            )
        return False
