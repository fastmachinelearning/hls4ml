"""Pack trained parameters into bank images, driven entirely by the manifest.

The core packer knows nothing about layers. It takes a flat sequence of scalars
and the structured ``layout`` the manifest declares, and produces memory words.

Getting from a tensor to that flat sequence is the only layer-specific step, and
the manifest describes it structurally in ``flat_order`` as an axis permutation,
so ``flatten`` stays generic too. A layer whose order is *not* a permutation of
its tensor axes (RNN gate ordering is the likely first case) will name an explicit
adapter in ``flat_order['adapter']``. No adapter is implemented yet, so such a
manifest is rejected rather than packed by the permutation path.

Nothing here infers a packing rule. A port the manifest declined to describe
cannot be packed, and a mismatch between the supplied data and the manifest is an
error rather than something to pad, truncate or transpose around.
"""

import json
import os
from fractions import Fraction

import numpy as np


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


def quantize_port(port, values):
    """Quantize a sequence of reals using the port's declared precision."""
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


def flatten(port, tensor):
    """Tensor -> flat scalar sequence, in the order the manifest declares.

    ``flat_order`` gives the tensor's own axis names, the shape it expects, and the
    order to enumerate them in, so this is a transpose followed by a ravel.
    """
    order = port.get('flat_order')
    if not order:
        raise PackingUnsupported(f'{port["name"]}: manifest declares no flat_order; cannot flatten')

    adapter = order.get('adapter')
    if adapter is not None:
        raise PackingUnsupported(
            f'{port["name"]}: manifest names flattener {adapter!r}, but no custom flatteners are implemented'
        )

    tensor_axes, axes = order['tensor_axes'], order['axes']
    if sorted(tensor_axes) != sorted(axes):
        raise PackingUnsupported(
            f'{port["name"]}: flat_order axes {axes} are not a permutation of {tensor_axes}; '
            'the manifest must name a flattener adapter for this layer'
        )

    array = np.asarray(tensor)
    expected_shape = tuple(order['shape'])
    if array.shape != expected_shape:
        raise PackingUnsupported(f'{port["name"]}: tensor shape {array.shape} but manifest declares {expected_shape}')

    permutation = [tensor_axes.index(axis) for axis in axes]
    return np.transpose(array, permutation).ravel().tolist()


def pack_flat(port, scalar_values):
    """Pack a flat scalar sequence into memory words, per the declared layout.

    ``scalar_values`` are real numbers; they are quantized once here using the
    port's declared precision. Returns a list of ints, one per word (or one per
    scalar port, for a fully partitioned bundle).
    """
    layout = port.get('layout')
    if not layout:
        raise PackingUnsupported(
            f'{port["name"]}: manifest declares no layout (kernel_variant={port.get("kernel_variant")}); cannot pack'
        )

    values = list(scalar_values)
    if len(values) != port['n_scalars']:
        raise PackingUnsupported(f'{port["name"]}: got {len(values)} scalars, manifest declares {port["n_scalars"]}')

    codes = quantize_port(port, values)

    mode = layout.get('mode')
    if mode == 'complete':
        # not a memory: one scalar port per element, so the "words" are the scalars
        return codes
    if mode != 'block':
        raise PackingUnsupported(f'{port["name"]}: layout mode {mode!r} is not supported')

    block_size = layout['block_size']
    width = port['precision']['width']
    words = [0] * port['expected_depth']
    for f, code in enumerate(codes):
        words[f % block_size] |= code << (width * (f // block_size))
    return words


def pack_tensor(port, tensor):
    """Convenience: flatten a tensor in the declared order, then pack it."""
    return pack_flat(port, flatten(port, tensor))


def write_mem(path, words, width_bits):
    """Emit a $readmemh image, one word per line, zero-padded to the full width."""
    digits = (width_bits + 3) // 4
    with open(path, 'w') as fh:
        for word in words:
            fh.write(f'{word:0{digits}x}\n')


def build_bank_image(port, banks, bank_stride_words=None):
    """Stack per-bank tensors into one depth-stacked memory image.

    BRAM ports only: a scalar bundle has no memory to stack into, and its banks are
    loaded scalar by scalar instead.
    """
    if port.get('expected_interface_kind') != 'bram':
        raise PackingUnsupported(
            f'{port["name"]}: build_bank_image is for bram ports; this port is '
            f'{port.get("expected_interface_kind")!r} and has no depth-stacked image'
        )

    per_bank = [pack_tensor(port, tensor) for tensor in banks]
    depth = port['expected_depth']
    stride = bank_stride_words or (1 << (depth - 1).bit_length())  # next power of two
    if stride < depth:
        raise PackingUnsupported(f'{port["name"]}: bank stride {stride} is smaller than depth {depth}')

    image = []
    for words in per_bank:
        image.extend(words)
        image.extend([0] * (stride - depth))  # padding words are never addressed
    return image, stride


def _parameter_inventory(project):
    """All (layer, role) parameter keys of a ModelGraph, or None for a bare path."""
    if not hasattr(project, 'get_layers'):
        return None
    return {(layer.name, role) for layer in project.get_layers() for role in getattr(layer, 'weights', {})}


def _load_manifest(project):
    """Read the external-parameter manifest a written project carries."""
    from hls4ml.writer.external_parameters import MANIFEST_FILENAME

    project_dir = project.config.get_output_dir() if hasattr(project, 'config') else str(project)
    path = os.path.join(project_dir, 'firmware', 'weights', MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise PackingUnsupported(
            f'no manifest at {path}; the project must be written with BramFactor set so that '
            'parameters are exposed outside the compute IP'
        )
    with open(path) as fh:
        return json.load(fh)


def pack_banks(project, banks):
    """Pack one complete parameter set per bank into per-port memory images.

    ``banks`` is a list of ``{(layer, role): tensor}``, one per bank, covering *every*
    parameter of the model. Only parameters ``BramFactor`` externalized may differ
    between them; the rest are compiled into the IP, so banks that disagree on a fixed
    parameter describe hardware that cannot be built and are rejected here.

    Tensors use hls4ml's declared layout: ``(n_in, n_out)`` for a Dense kernel, the
    Keras orientation. A PyTorch ``nn.Linear.weight`` is its transpose. Shapes are
    checked exactly, so only a square kernel could pass unnoticed.

    ``project`` is a ModelGraph or a written project path; a ModelGraph also gives the
    full parameter inventory, which is what lets a *missing* fixed parameter be
    reported rather than ignored.

    Returns ``{port_name: {...}}``: BRAM ports carry ``image`` and
    ``bank_stride_words``, scalar bundles per-bank ``codes``.
    """
    if len(banks) < 2:
        raise PackingUnsupported(f'need at least 2 banks, got {len(banks)}')

    manifest = _load_manifest(project)
    inventory = _parameter_inventory(project)
    external = {(p['layer'], p['role']): p for p in manifest['ports']}

    missing = [k for k in external for bank in banks if k not in bank]
    if missing:
        raise PackingUnsupported(f'every bank must supply each externalized parameter; missing {sorted(set(missing))}')

    supplied = {k for bank in banks for k in bank}
    if inventory is not None:
        unknown = supplied - inventory
        if unknown:
            raise PackingUnsupported(f'not parameters of this model: {sorted(unknown)}')
        # Without the full set there is no way to tell a fixed parameter that
        # agrees from one that was simply never supplied.
        absent = inventory - supplied
        if absent:
            raise PackingUnsupported(
                f'each bank must be a complete parameter set so the fixed parameters can be '
                f'checked; missing {sorted(absent)}'
            )

    # A fixed parameter lives in the compute IP and cannot vary per bank.
    differing = []
    for key in sorted(supplied - set(external), key=str):
        values = [bank.get(key) for bank in banks]
        first = np.asarray(values[0])
        if any(v is None for v in values) or any(not np.array_equal(first, np.asarray(v)) for v in values[1:]):
            differing.append(key)
    if differing:
        raise PackingUnsupported(
            f'{differing} are not externalized by BramFactor, so they are fixed in the compute IP, '
            'but the supplied banks disagree on them; every bank must share the same fixed parameters'
        )

    packed = {}
    for key, port in external.items():
        tensors = [bank[key] for bank in banks]
        if port['expected_interface_kind'] == 'bram':
            image, stride = build_bank_image(port, tensors)
            packed[port['name']] = {
                'kind': 'bram',
                'image': image,
                'bank_stride_words': stride,
                'data_width': port['expected_data_width'],
            }
        else:
            packed[port['name']] = {
                'kind': port['expected_interface_kind'],
                'codes': [pack_flat(port, np.asarray(t).ravel().tolist()) for t in tensors],
                'data_width': port['expected_data_width'],
            }
    return packed
