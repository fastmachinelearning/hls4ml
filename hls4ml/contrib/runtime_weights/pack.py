"""Pack trained parameters into bank images, driven entirely by the manifest.

The packing rule is never assumed here: it is read from the manifest's
``reshape.lane_of_flat_index`` / ``word_of_flat_index``, which hls4ml only
populates for kernel variants whose mapping has been verified against generated
RTL. A port the manifest declined to describe cannot be packed.
"""

import re
from fractions import Fraction


class PackingUnsupported(Exception):
    """The manifest does not describe this port well enough to pack it."""


def quantize(value, width, integer, signed=True, rounding='TRN', saturation='WRAP'):
    """Real number -> raw two's-complement code for an ap_fixed<width,integer>.

    Only the combination hls4ml's default precision uses (AP_TRN / AP_WRAP) is
    implemented; anything else raises rather than packing silently-wrong values.
    """
    if rounding not in ('TRN',) or saturation not in ('WRAP',):
        raise PackingUnsupported(f'quantization {rounding}/{saturation} not implemented; only TRN/WRAP is supported')
    frac_bits = width - integer
    scaled = Fraction(value).limit_denominator(1 << 30) * (1 << frac_bits)
    code = scaled.numerator // scaled.denominator  # AP_TRN: floor toward -inf
    return code % (1 << width)  # AP_WRAP


def _affine(expr, name='f'):
    """Parse 'f // 2' or 'f % 2' -> ('//', 2) / ('%', 2)."""
    match = re.fullmatch(rf'\s*{name}\s*(//|%)\s*(\d+)\s*', expr or '')
    if not match:
        raise PackingUnsupported(f'cannot parse index expression {expr!r}')
    return match.group(1), int(match.group(2))


def flat_index(port, in_index, out_index, n_in):
    """Logical (in, out) -> hls4ml flat scalar index, per the manifest's order."""
    order = port['logical_scalar_order']
    if order == 'index(out,in) = out*n_in + in':
        return out_index * n_in + in_index
    raise PackingUnsupported(f'unknown logical scalar order {order!r}')


def pack_weight_bank(port, weights, n_in, n_out):
    """Pack one bank of a 2-D weight tensor into physical memory words.

    ``weights`` is indexed [in][out]. Returns a list of ints, one per word.
    """
    if port['expected_interface_kind'] != 'bram':
        raise PackingUnsupported(f'{port["name"]} is not a bram port')
    reshape = port.get('reshape') or {}
    if not reshape.get('lane_of_flat_index'):
        raise PackingUnsupported(
            f'{port["name"]}: manifest claims no lane/word mapping '
            f'(kernel_variant={port.get("kernel_variant")}); cannot pack'
        )

    lane_op, lane_div = _affine(reshape['lane_of_flat_index'])
    word_op, word_div = _affine(reshape['word_of_flat_index'])
    if lane_op != '//' or word_op != '%':
        raise PackingUnsupported('unexpected lane/word operators in manifest')

    precision = port['precision']
    scalar_width = precision['width']
    depth = port['expected_depth']
    words = [0] * depth

    for i in range(n_in):
        for o in range(n_out):
            f = flat_index(port, i, o, n_in)
            lane = f // lane_div
            word = f % word_div
            code = quantize(
                weights[i][o],
                scalar_width,
                precision['integer'],
                precision['signed'],
                precision.get('rounding_mode', 'TRN'),
                precision.get('saturation_mode', 'WRAP'),
            )
            words[word] |= code << (scalar_width * lane)
    return words


def pack_scalar_bank(port, values):
    """Pack a scalar-bundle parameter (e.g. a Dense bias) into raw codes."""
    if port['expected_interface_kind'] != 'scalar_bundle':
        raise PackingUnsupported(f'{port["name"]} is not a scalar bundle')
    precision = port['precision']
    return [
        quantize(
            v,
            precision['width'],
            precision['integer'],
            precision['signed'],
            precision.get('rounding_mode', 'TRN'),
            precision.get('saturation_mode', 'WRAP'),
        )
        for v in values
    ]


def write_mem(path, words, width_bits):
    """Emit a $readmemh image, one word per line, zero-padded to the full width."""
    digits = (width_bits + 3) // 4
    with open(path, 'w') as fh:
        for word in words:
            fh.write(f'{word:0{digits}x}\n')


def build_bank_image(port, banks, n_in, n_out, bank_stride_words=None):
    """Stack per-bank weight images into one physical image, padding to the stride."""
    per_bank = [pack_weight_bank(port, w, n_in, n_out) for w in banks]
    depth = port['expected_depth']
    stride = bank_stride_words or (1 << (depth - 1).bit_length())  # next power of two
    image = []
    for words in per_bank:
        image.extend(words)
        image.extend([0] * (stride - depth))  # padding words are never addressed
    return image, stride
