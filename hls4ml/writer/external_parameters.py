"""Machine-readable description of external parameter ports.

When ``BramFactor`` exposes parameters outside the compute IP, a consumer that
wants to populate them needs to know two things that only hls4ml knows: the order
in which it flattened each tensor, and the ``ARRAY_RESHAPE`` its layer templates
emitted. This module writes both to
``firmware/weights/external_parameters.json``.

It is *not* only about BRAM: a fully partitioned parameter (a Dense bias, say)
becomes a bundle of scalar ports rather than a memory, and is described here too.

Design contract
---------------
hls4ml owns the *logical* scalar ordering and the pragma it emitted. **HLS owns
the physical interface.** Every geometry field is therefore named ``expected_*``:
it states what hls4ml asked for, not what was built. Only C/RTL synthesis or
export establishes the actual geometry, and consumers must cross-check.

Scope
-----
Descriptions come from a registry keyed by
``(backend, io_type, strategy, layer_class, role)``. The writer sees *every*
external parameter, including layers whose packing has never been verified, so an
unregistered combination gets **no** interface kind, geometry or ordering -- only
a note telling the consumer to classify from the export report. Silence is the
default; a guess is never emitted.
"""

import json
import os

SCHEMA = 'hls4ml.external_parameter_manifest/v1'
SCHEMA_VERSION = 1

MANIFEST_FILENAME = 'external_parameters.json'

# (backend, io_type, strategy, layer_class, role) -> describe(context) -> dict
# strategy is compared lower-case; backend and layer_class keep hls4ml's casing.
_REGISTRY = {}


def register(backend, io_type, strategy, layer_class, role):
    """Register a description adapter for one parameter kind."""

    def decorator(func):
        _REGISTRY[(backend, io_type, strategy.lower(), layer_class, role)] = func
        return func

    return decorator


def registered_keys():
    """Every (backend, io_type, strategy, layer_class, role) currently described."""
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Dense, Vitis, io_parallel, Resource
# --------------------------------------------------------------------------

# Kernel variants whose reshape -> (lane, word) mapping is verified against
# generated RTL and tool-generated co-simulation memory images.
#
# dense_resource_rf_gt_nin is deliberately absent: it is unreachable for Dense.
# nnet_dense_resource.h dispatches to it when (rf > n_in and rf % n_in != 0), but
# FPGABackend._validate_reuse_factor asserts ((rf % n_in) == 0) or (rf < n_in) --
# the exact negation -- and init_dense snaps offending values via
# set_closest_reuse_factor.
VERIFIED_DENSE_KERNELS = {
    'dense_resource_rf_leq_nin',
    'dense_resource_rf_gt_nin_rem0',
}


# Quantization a packing consumer can actually encode. A layout is claimed only
# for these; anything else (integer types, rounding/saturation variants) would
# promise a mapping that cannot be reproduced, so it is described but unclaimed.
SUPPORTED_ROUNDING = {'TRN'}
SUPPORTED_SATURATION = {'WRAP'}


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


@register('Vitis', 'io_parallel', 'Resource', 'Dense', 'weight')
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

    described.update(
        expected_interface_kind='bram',
        expected_data_width=block_factor * ctx['precision']['width'],
        expected_depth=block_size,
        flat_order={
            'tensor_axes': ['n_in', 'n_out'],
            'axes': ['n_out', 'n_in'],
            'shape': [n_in, n_out],
            'description': 'index(out,in) = out*n_in + in',
        },
        layout={
            'mode': 'block',
            'block_size': block_size,
            'lanes': block_factor,
            'description': f'lane = f // {block_size}, word = f % {block_size}',
        },
    )
    return described


@register('Vitis', 'io_parallel', 'Resource', 'Dense', 'bias')
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
            'description': 'b[i] = bias[i], i = out index',
        },
        'layout': {'mode': 'complete', 'description': 'one scalar port per element'},
    }


# --------------------------------------------------------------------------


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
        describe = _REGISTRY.get(key)
        if describe is None:
            entry['note'] = (
                f'no adapter registered for {key}; no interface kind, geometry or '
                'ordering is claimed -- classify from the export report'
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
