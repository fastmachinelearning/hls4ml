"""Machine-readable description of external BRAM parameter ports.

When ``BramFactor`` exposes weights as external ``#pragma HLS INTERFACE bram``
ports, a consumer that wants to populate those ports needs to know which logical
scalar lands in which physical (word, lane). That mapping follows from the
``#pragma HLS ARRAY_RESHAPE ... block factor=N`` emitted by hls4ml's own layer
templates, but is not otherwise recorded anywhere.

This module emits ``firmware/weights/bram_manifest.json`` describing it.

Design contract
---------------
hls4ml owns the *logical* scalar ordering and the pragma it emitted. **HLS owns
the physical interface.** Every geometry field is therefore named ``expected_*``:
it states what hls4ml asked for, not what was built. Only C/RTL synthesis or
export establishes the actual geometry, and consumers must cross-check before
relying on it.

Scope
-----
``SUPPORTED_SCOPE`` is an allowlist. The writer iterates over *every* weight
variable with ``storage == 'bram'``, which may include layers whose packing has
never been verified. Anything outside the allowlist gets **no** interface kind,
geometry or ordering -- only a note telling the consumer to classify from the
export report. Silence is the default; a guess is never emitted.
"""

import json
import os

SCHEMA = 'hls4ml.bram_manifest/v1'
SCHEMA_VERSION = 1

# Backends whose generated interface has been verified against the manifest.
# The Vivado backend also implements BramFactor but has not been checked, so it
# gets no claims.
SUPPORTED_BACKENDS = {'Vitis'}

# (layer_class, weight role) pairs whose packing has been verified against
# generated RTL and tool-generated co-simulation memory images. Additionally
# gated on backend, Strategy == Resource and io_type == io_parallel.
SUPPORTED_SCOPE = {
    ('Dense', 'weight'),
    ('Dense', 'bias'),
}

# Kernel variants whose reshape -> (lane, word) mapping is verified.
#
# All three nnet_dense_resource.h variants emit an identical weights pragma
# (``ARRAY_RESHAPE variable=weights block factor=block_factor``), but only the
# two reachable ones are claimed here.
#
# dense_resource_rf_gt_nin is deliberately absent: it is unreachable for Dense.
# The template dispatches to it when (rf > n_in and rf % n_in != 0), but
# FPGABackend._validate_reuse_factor asserts ((rf % n_in) == 0) or (rf < n_in) --
# the exact negation -- and init_dense snaps offending values via
# set_closest_reuse_factor.
VERIFIED_KERNELS = {
    'dense_resource_rf_leq_nin',
    'dense_resource_rf_gt_nin_rem0',
}


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


def _dense_kernel_variant(n_in, reuse_factor):
    """Mirror the dispatch in nnet_dense_resource.h::dense_resource."""
    if n_in is None or reuse_factor is None:
        return None
    if reuse_factor <= n_in:
        return 'dense_resource_rf_leq_nin'
    if reuse_factor % n_in == 0:
        return 'dense_resource_rf_gt_nin_rem0'
    return 'dense_resource_rf_gt_nin'


def _weight_reshape(kernel, n_scalars, reuse_factor):
    """Describe the ARRAY_RESHAPE hls4ml emitted, and the resulting lane/word map.

    ``block factor=N`` on an array of size S gives block_size = ceil(S / N); element
    f lands in block (f // block_size) at offset (f % block_size). The reshape
    concatenates the N blocks into one word per offset, so lane = f // block_size
    and word = f % block_size.
    """
    block_factor = -(-n_scalars // reuse_factor) if reuse_factor else None
    block_size = -(-n_scalars // block_factor) if block_factor else None

    info = {
        'pragma': f'ARRAY_RESHAPE variable=weights block factor={block_factor}',
        'mode': 'block',
        'factor': block_factor,
        'block_size': block_size,
        'lane_of_flat_index': None,
        'word_of_flat_index': None,
        'verified_for': [],
    }
    if kernel in VERIFIED_KERNELS and block_size:
        info['lane_of_flat_index'] = f'f // {block_size}'
        info['word_of_flat_index'] = f'f % {block_size}'
        info['verified_for'] = [kernel]
    else:
        info['note'] = f"kernel variant '{kernel}' is not in VERIFIED_KERNELS; lane/word mapping intentionally omitted"
    return info


def build_bram_manifest(model):
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
            'logical_scalar_order': None,
            'kernel_variant': None,
            'reshape': None,
            'expected_interface_kind': None,
            'expected_data_width': None,
            'expected_depth': None,
            'values_txt': f'firmware/weights/{var.name}.txt' if write_txt else None,
        }

        in_scope = (
            layer is not None
            and backend in SUPPORTED_BACKENDS
            and (layer.class_name, role) in SUPPORTED_SCOPE
            and str(strategy).lower() == 'resource'
            and io_type == 'io_parallel'
        )

        if not in_scope:
            entry['note'] = (
                f'outside schema v1 scope (backend={backend}, layer_class={entry["layer_class"]}, '
                f'role={role}, strategy={strategy}, io_type={io_type}); no interface kind, geometry or '
                'ordering is claimed -- classify from the export report'
            )
        elif role == 'weight':
            kernel = _dense_kernel_variant(layer.get_attr('n_in'), reuse_factor)
            reshape = _weight_reshape(kernel, n_scalars, reuse_factor)
            entry['logical_scalar_order'] = 'index(out,in) = out*n_in + in'
            entry['kernel_variant'] = kernel
            entry['reshape'] = reshape
            entry['expected_interface_kind'] = 'bram'
            if reshape['factor'] and precision['width']:
                entry['expected_data_width'] = reshape['factor'] * precision['width']
                entry['expected_depth'] = reshape['block_size']
        else:  # bias
            # Every dense_resource variant carries
            # '#pragma HLS ARRAY_PARTITION variable=biases complete', which wins over
            # the BRAM interface, so a bias lowers to scalar ap_none ports regardless
            # of size. True for the templates and tool flow tested; it follows from a
            # template pragma and is not a permanent property.
            entry['logical_scalar_order'] = 'b[i] = bias[i], i = out index'
            entry['reshape'] = {
                'pragma': 'ARRAY_PARTITION variable=biases complete',
                'mode': 'complete',
                'verified_for': sorted(VERIFIED_KERNELS),
            }
            entry['expected_interface_kind'] = 'scalar_bundle'
            entry['expected_data_width'] = precision['width']

        ports.append(entry)

    return {
        'schema': SCHEMA,
        'schema_version': SCHEMA_VERSION,
        'hls4ml_version': hls4ml_version,
        'project_name': config.get_project_name(),
        'backend': str(config.get_config_value('Backend')),
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


def write_bram_manifest(model, path=None):
    """Write the manifest. Returns its path, or None when there is nothing to describe."""
    manifest = build_bram_manifest(model)
    if not manifest['ports']:
        return None

    if path is None:
        path = os.path.join(model.config.get_output_dir(), 'firmware', 'weights', 'bram_manifest.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    return path
