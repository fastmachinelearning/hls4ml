"""Describe parameters exposed outside the HLS compute IP.

The manifest records logical tensor ordering and the expected external interface
geometry. Synthesized RTL remains authoritative and is verified separately.
Unsupported layer/interface combinations are left unclaimed.
"""

import json
import os

SCHEMA = 'hls4ml.external_parameter_manifest/v1'
SCHEMA_VERSION = 1

MANIFEST_FILENAME = 'external_parameters.json'


# dense_resource_rf_gt_nin is intentionally unsupported: hls4ml's reuse-factor
# validation makes this Dense kernel variant unreachable.
VERIFIED_DENSE_KERNELS = {
    'dense_resource_rf_leq_nin',
    'dense_resource_rf_gt_nin_rem0',
}


# Quantization a packing consumer can actually encode. A layout is claimed only
# for these for now; anything else (int types, rounding/saturation variants) not verified.
SUPPORTED_ROUNDING = {'TRN'}
SUPPORTED_SATURATION = {'WRAP'}

# A reshaped weight array is only an addressable memory if it has more than one
# word. reuse_factor == 1 collapses the whole array into a single word, which has
# no address to concatenate a bank id onto, and which HLS does not build as asked.
MIN_ADDRESSABLE_DEPTH = 2

# Scope boundary for schema v1, not a tool limit: reshaped words up to 4096 bits
# are verified end-to-end. 4096 is an established HLS array-partition threshold,
# and wider reshaped interfaces have no trustworthy packing contract here yet, so
# they are refused rather than guessed.
MAX_RESHAPED_PORT_BITS = 4096


def _unsupported_precision_reason(precision):
    """Return why this precision cannot be packed, or None if it can."""
    if precision.get('width') is None or precision.get('integer') is None:
        return f'precision {precision.get("type")!r} has no width/integer bits'
    rounding = precision.get('rounding_mode')
    saturation = precision.get('saturation_mode')
    if rounding not in SUPPORTED_ROUNDING or saturation not in SUPPORTED_SATURATION:
        return (
            f'precision {precision.get("type")!r} uses rounding={rounding}/saturation={saturation}; '
            f'only {sorted(SUPPORTED_ROUNDING)}/{sorted(SUPPORTED_SATURATION)} can be encoded'
        )
    return None


def _dense_kernel_variant(n_in, reuse_factor):
    """Mirror the dispatch in nnet_dense_resource.h::dense_resource."""
    if n_in is None or reuse_factor is None:
        return None
    if reuse_factor <= n_in:
        return 'dense_resource_rf_leq_nin'
    if reuse_factor % n_in == 0:
        return 'dense_resource_rf_gt_nin_rem0'
    return 'dense_resource_rf_gt_nin'


def _describe_dense_weight(ctx):
    """Dense kernel: reshaped by `ARRAY_RESHAPE variable=weights block factor=N`.

    ``block factor=N`` on an array of size S gives block_size = ceil(S / N); element
    f lands in block (f // block_size) at offset (f % block_size). The reshape
    concatenates the N blocks into one word per offset, so a word holds N lanes and
    the memory is block_size words deep.
    """
    n_scalars, reuse_factor = ctx['n_scalars'], ctx['reuse_factor']
    n_in = ctx['layer'].get_attr('n_in')
    n_out = ctx['layer'].get_attr('n_out')
    kernel = _dense_kernel_variant(n_in, reuse_factor)

    block_factor = -(-n_scalars // reuse_factor) if reuse_factor else None
    block_size = -(-n_scalars // block_factor) if block_factor else None

    described = {
        'kernel_variant': kernel,
        'pragma': f'ARRAY_RESHAPE variable=weights block factor={block_factor}',
    }
    unsupported = _unsupported_precision_reason(ctx['precision'])
    if unsupported:
        described['note'] = f'{unsupported}; no layout or ordering is claimed'
        return described
    if kernel not in VERIFIED_DENSE_KERNELS or not block_size:
        described['note'] = f"kernel variant '{kernel}' is not verified; no layout or ordering is claimed"
        return described
    if block_size < MIN_ADDRESSABLE_DEPTH:
        described['note'] = (
            f'reuse_factor={reuse_factor} reshapes all {n_scalars} scalars into a single word, so the '
            'port has no address to bank. Raise the reuse factor (the memory depth equals it) to make '
            'this parameter bankable.'
        )
        return described
    port_bits = block_factor * ctx['precision']['width']
    if port_bits > MAX_RESHAPED_PORT_BITS:
        described['note'] = (
            f'reshaped port would be {port_bits} bits, above the {MAX_RESHAPED_PORT_BITS}-bit word this '
            'schema verifies; raise the reuse factor to narrow the word'
        )
        return described

    described.update(
        expected_interface_kind='bram',
        expected_data_width=block_factor * ctx['precision']['width'],
        expected_depth=block_size,
        flat_order={
            'tensor_axes': ['n_in', 'n_out'],
            'axes': ['n_out', 'n_in'],
            'shape': [n_in, n_out],
        },
        layout={
            'mode': 'block',
            'block_size': block_size,
            'lanes': block_factor,
        },
    )
    return described


def _describe_dense_bias(ctx):
    """Dense bias: `ARRAY_PARTITION variable=biases complete` wins over the BRAM
    interface, so it lowers to scalar ports regardless of size.

    Scoped claim: true for the templates and tool flow tested. It follows from a
    template pragma and is not a permanent property.
    """
    unsupported = _unsupported_precision_reason(ctx['precision'])
    if unsupported:
        return {
            'pragma': 'ARRAY_PARTITION variable=biases complete',
            'note': f'{unsupported}; no layout or ordering is claimed',
        }
    return {
        'expected_interface_kind': 'scalar_bundle',
        'expected_data_width': ctx['precision']['width'],
        'expected_depth': None,
        'pragma': 'ARRAY_PARTITION variable=biases complete',
        'flat_order': {
            'tensor_axes': ['n_out'],
            'axes': ['n_out'],
            'shape': [ctx['n_scalars']],
        },
        'layout': {'mode': 'complete'},
    }


def _describe_pointwise_weight(ctx):
    """PointwiseConv kernel: not reshaped at the external port.

    The pointwise path buffers weights internally before the dense multiply, so the
    ``ARRAY_RESHAPE`` never reaches the interface. HLS exposes a plain memory one
    scalar wide and ``n_chan * n_filt`` deep, independent of the reuse factor
    (verified across reuse factors 1/2/8 -- including 1 -- and several channel and
    filter counts).

    hls4ml declares the kernel as ``(filt..., n_chan, n_filt)`` and stores it
    filter-major. A Dense over 2-D/3-D input and a native ``Conv*D`` with a 1-wide
    kernel give the same layer with the same declared shape, so this describes the
    class, not one origin.
    """
    layer = ctx['layer']
    n_chan = layer.get_attr('n_chan')
    n_filt = layer.get_attr('n_filt')
    filt_width = layer.get_attr('filt_width')
    filt_height = layer.get_attr('filt_height')
    two_d = filt_height is not None

    described = {'kernel_variant': 'pointwise_unreshaped'}

    unsupported = _unsupported_precision_reason(ctx['precision'])
    if unsupported:
        described['note'] = f'{unsupported}; no layout or ordering is claimed'
        return described
    # The class name already implies a 1-wide kernel; check it rather than trust it,
    # since the layout below is only correct for that.
    if filt_width != 1 or (two_d and filt_height != 1):
        described['note'] = f'filter is {filt_height}x{filt_width}, not 1-wide; the pointwise layout does not apply'
        return described
    if layer.get_attr('implementation') != 'linebuffer':
        described['note'] = (
            f'conv implementation is {layer.get_attr("implementation")!r}, but only linebuffer has been verified'
        )
        return described

    n_scalars = ctx['n_scalars']
    if n_scalars != n_chan * n_filt:
        described['note'] = f'{n_scalars} scalars is not n_chan*n_filt ({n_chan}*{n_filt}); layout unclear'
        return described
    if n_scalars < MIN_ADDRESSABLE_DEPTH:
        described['note'] = f'{n_scalars} scalars leaves no address to bank'
        return described

    if two_d:
        tensor_axes = ['filt_height', 'filt_width', 'n_chan', 'n_filt']
        shape = [1, 1, n_chan, n_filt]
        axes = ['n_filt', 'filt_height', 'filt_width', 'n_chan']
    else:
        tensor_axes = ['filt_width', 'n_chan', 'n_filt']
        shape = [1, n_chan, n_filt]
        axes = ['n_filt', 'filt_width', 'n_chan']

    described.update(
        expected_interface_kind='bram',
        expected_data_width=ctx['precision']['width'],  # one scalar per word
        expected_depth=n_scalars,
        flat_order={'tensor_axes': tensor_axes, 'axes': axes, 'shape': shape},
        layout={'mode': 'block', 'block_size': n_scalars, 'lanes': 1},
    )
    return described


def _describe_pointwise_bias(ctx):
    """PointwiseConv bias: one scalar port per filter, as for a Dense bias."""
    unsupported = _unsupported_precision_reason(ctx['precision'])
    if unsupported:
        return {'note': f'{unsupported}; no layout or ordering is claimed'}
    return {
        'expected_interface_kind': 'scalar_bundle',
        'expected_data_width': ctx['precision']['width'],
        'expected_depth': None,
        'flat_order': {'tensor_axes': ['n_filt'], 'axes': ['n_filt'], 'shape': [ctx['n_scalars']]},
        'layout': {'mode': 'complete'},
    }


# (backend, io_type, strategy, layer_class, role) -> describe(context) -> dict.
# strategy is matched lower-case; backend and layer_class keep hls4ml's casing.
# Adding an entry is the only way to widen the manifest's scope, and requires
# evidence that the packing has been verified against generated RTL.
_ADAPTERS = {
    ('Vitis', 'io_parallel', 'resource', 'Dense', 'weight'): _describe_dense_weight,
    ('Vitis', 'io_parallel', 'resource', 'Dense', 'bias'): _describe_dense_bias,
    ('Vitis', 'io_parallel', 'resource', 'PointwiseConv1D', 'weight'): _describe_pointwise_weight,
    ('Vitis', 'io_parallel', 'resource', 'PointwiseConv1D', 'bias'): _describe_pointwise_bias,
    ('Vitis', 'io_parallel', 'resource', 'PointwiseConv2D', 'weight'): _describe_pointwise_weight,
    ('Vitis', 'io_parallel', 'resource', 'PointwiseConv2D', 'bias'): _describe_pointwise_bias,
}


def described_combinations():
    """Every (backend, io_type, strategy, layer_class, role) the manifest describes."""
    return sorted(_ADAPTERS)


def _precision_dict(precision):
    width = getattr(precision, 'width', None)
    integer = getattr(precision, 'integer', None)
    signed = getattr(precision, 'signed', None)
    out = {
        'type': str(precision),
        'width': width,
        'integer': integer,
        'fractional': (width - integer) if (width is not None and integer is not None) else None,
        'signed': bool(signed) if signed is not None else None,
    }
    for attr in ('rounding_mode', 'saturation_mode', 'saturation_bits'):
        value = getattr(precision, attr, None)
        if value is not None:
            out[attr] = str(value)
    return out


def _owning_layer(model, var):
    """Return (layer, role) for a weight variable; role is its key in layer.weights."""
    for layer in model.get_layers():
        for role, weight_var in getattr(layer, 'weights', {}).items():
            if weight_var is var or getattr(weight_var, 'name', None) == var.name:
                return layer, role
    return None, None


def build_manifest(model):
    """Return the manifest dict for a ModelGraph ('ports' is empty if none apply)."""
    config = model.config
    writer_config = config.get_writer_config() or {}
    write_txt = bool(writer_config.get('WriteWeightsTxt', False))
    io_type = config.get_config_value('IOType')
    backend = str(config.get_config_value('Backend'))

    try:
        from hls4ml import __version__ as hls4ml_version
    except ImportError:  # pragma: no cover
        hls4ml_version = None

    ports = []
    for var in model.get_weight_variables():
        if str(getattr(var, 'storage', '')).lower() != 'bram':
            continue

        layer, role = _owning_layer(model, var)
        n_scalars = int(getattr(var, 'data_length', 0) or 0)
        reuse_factor = layer.get_attr('reuse_factor') if layer else None
        strategy = layer.get_attr('strategy') if layer else None
        precision = _precision_dict(var.type.precision)

        entry = {
            'name': var.name,
            'layer': layer.name if layer else None,
            'layer_class': layer.class_name if layer else None,
            'role': role,
            'tensor_shape': list(getattr(var, 'shape', []) or []),
            'n_scalars': n_scalars,
            'precision': precision,
            'reuse_factor': reuse_factor,
            'strategy': str(strategy) if strategy is not None else None,
            'kernel_variant': None,
            'flat_order': None,
            'layout': None,
            'expected_interface_kind': None,
            'expected_data_width': None,
            'expected_depth': None,
            'values_txt': f'firmware/weights/{var.name}.txt' if write_txt else None,
        }

        key = (backend, io_type, str(strategy).lower(), entry['layer_class'], role)
        describe = _ADAPTERS.get(key)
        if describe is None:
            entry['note'] = (
                f'no adapter for {key}; no interface kind, geometry or ordering is claimed -- '
                'classify from the export report. Note that a fully partitioned parameter '
                '(Strategy=Latency, or any ARRAY_PARTITION complete) has no address to bank: '
                'it lowers to one port per element, so it is out of scope by construction '
                'rather than by omission.'
            )
        else:
            entry.update(
                describe(
                    {
                        'layer': layer,
                        'var': var,
                        'n_scalars': n_scalars,
                        'reuse_factor': reuse_factor,
                        'precision': precision,
                    }
                )
            )

        ports.append(entry)

    return {
        'schema': SCHEMA,
        'schema_version': SCHEMA_VERSION,
        'hls4ml_version': hls4ml_version,
        'project_name': config.get_project_name(),
        'backend': backend,
        'io_type': io_type,
        'part': config.get_config_value('Part'),
        'clock_period': config.get_config_value('ClockPeriod'),
        'bram_factor': getattr(config, 'model_bf', None),
        'disclaimer': (
            "All 'expected_*' fields state what hls4ml requested via pragmas. Only C/RTL "
            'synthesis or export establishes the actual interface geometry. Consumers must '
            'cross-check before relying on them.'
        ),
        'ports': ports,
    }


def write_manifest(model, path=None):
    """Write the manifest. Returns its path, or None when there is nothing to describe."""
    manifest = build_manifest(model)
    if not manifest['ports']:
        return None

    if path is None:
        path = os.path.join(model.config.get_output_dir(), 'firmware', 'weights', MANIFEST_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    return path
