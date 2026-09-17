"""Describe parameters exposed outside the HLS compute IP.

Adapters define the expected interface and packing layout for supported
layer/parameter combinations. Synthesized RTL is verified separately.
"""

from hls4ml.model.external_parameters import (
    BlockLayout,
    CompleteLayout,
    ExternalParameter,
    ExternalParameterManifest,
    FlatOrder,
    unencodable_reason,
)

# dense_resource_rf_gt_nin is intentionally unsupported: hls4ml's reuse-factor
# validation makes this Dense kernel variant unreachable.
VERIFIED_DENSE_KERNELS = {
    'dense_resource_rf_leq_nin',
    'dense_resource_rf_gt_nin_rem0',
}


# A reshaped weight array is only an addressable memory if it has more than one
# word. reuse_factor == 1 collapses the whole array into a single word, which has
# no address to concatenate a bank id onto, and which HLS does not build as asked.
MIN_ADDRESSABLE_DEPTH = 2

# Scope boundary for schema v1, not a tool limit: reshaped words up to 4096 bits
# are verified end-to-end. 4096 is an established HLS array-partition threshold,
# and wider reshaped interfaces have no trustworthy packing contract here yet, so
# they are refused rather than guessed.
MAX_RESHAPED_PORT_BITS = 4096


def _dense_kernel_variant(n_in, reuse_factor):
    """Mirror the dispatch in nnet_dense_resource.h::dense_resource."""
    if n_in is None or reuse_factor is None:
        return None
    if reuse_factor <= n_in:
        return 'dense_resource_rf_leq_nin'
    if reuse_factor % n_in == 0:
        return 'dense_resource_rf_gt_nin_rem0'
    return 'dense_resource_rf_gt_nin'


def _describe_dense_weight(layer, n_scalars, reuse_factor, precision):
    """Dense kernel: reshaped by `ARRAY_RESHAPE variable=weights block factor=N`.

    ``block factor=N`` on an array of size S gives block_size = ceil(S / N); element
    f lands in block (f // block_size) at offset (f % block_size). The reshape
    concatenates the N blocks into one word per offset, so a word holds N lanes and
    the memory is block_size words deep.
    """
    n_in = layer.get_attr('n_in')
    n_out = layer.get_attr('n_out')
    kernel = _dense_kernel_variant(n_in, reuse_factor)

    block_factor = -(-n_scalars // reuse_factor) if reuse_factor else None
    block_size = -(-n_scalars // block_factor) if block_factor else None

    described = {
        'kernel_variant': kernel,
        'pragma': f'ARRAY_RESHAPE variable=weights block factor={block_factor}',
    }
    unsupported = unencodable_reason(precision)
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
    port_bits = block_factor * precision.width
    if port_bits > MAX_RESHAPED_PORT_BITS:
        described['note'] = (
            f'reshaped port would be {port_bits} bits, above the {MAX_RESHAPED_PORT_BITS}-bit word this '
            'schema verifies; raise the reuse factor to narrow the word'
        )
        return described

    described.update(
        interface_kind='bram',
        data_width=port_bits,
        depth=block_size,
        flat_order=FlatOrder(['n_in', 'n_out'], ['n_out', 'n_in'], [n_in, n_out]),
        layout=BlockLayout(block_size, block_factor),
    )
    return described


def _describe_dense_bias(layer, n_scalars, reuse_factor, precision):
    """Describe a Dense bias lowered to scalar ports by complete partitioning."""
    pragma = 'ARRAY_PARTITION variable=biases complete'
    unsupported = unencodable_reason(precision)
    if unsupported:
        return {'pragma': pragma, 'note': f'{unsupported}; no layout or ordering is claimed'}
    return {
        'pragma': pragma,
        'interface_kind': 'scalar_bundle',
        'data_width': precision.width,
        'flat_order': FlatOrder(['n_out'], ['n_out'], [n_scalars]),
        'layout': CompleteLayout(),
    }


def _describe_pointwise_weight(layer, n_scalars, reuse_factor, precision):
    """Describe the unreshaped external interface of a pointwise kernel.

    Pointwise weights reach the interface one scalar per word, independent of
    reuse factor. Dense layers over higher-rank inputs and native 1-wide
    convolutions use the same pointwise layer representation.
    """
    n_chan = layer.get_attr('n_chan')
    n_filt = layer.get_attr('n_filt')
    filt_width = layer.get_attr('filt_width')
    filt_height = layer.get_attr('filt_height')
    two_d = filt_height is not None

    described = {'kernel_variant': 'pointwise_unreshaped'}

    unsupported = unencodable_reason(precision)
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
        interface_kind='bram',
        data_width=precision.width,  # one scalar per word
        depth=n_scalars,
        flat_order=FlatOrder(tensor_axes, axes, shape),
        layout=BlockLayout(n_scalars, 1),
    )
    return described


def _describe_pointwise_bias(layer, n_scalars, reuse_factor, precision):
    """PointwiseConv bias: one scalar port per filter, as for a Dense bias."""
    unsupported = unencodable_reason(precision)
    if unsupported:
        return {'note': f'{unsupported}; no layout or ordering is claimed'}
    return {
        'interface_kind': 'scalar_bundle',
        'data_width': precision.width,
        'flat_order': FlatOrder(['n_filt'], ['n_filt'], [n_scalars]),
        'layout': CompleteLayout(),
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


def _owning_layer(model, var):
    """Return (layer, role) for a weight variable; role is its key in layer.weights."""
    for layer in model.get_layers():
        for role, weight_var in getattr(layer, 'weights', {}).items():
            if weight_var is var or getattr(weight_var, 'name', None) == var.name:
                return layer, role
    return None, None


def describe_parameter(model, var):
    """The ExternalParameter for one weight variable the backend exposed."""
    config = model.config
    io_type = config.get_config_value('IOType')
    backend = str(config.get_config_value('Backend'))

    layer, role = _owning_layer(model, var)
    n_scalars = int(getattr(var, 'data_length', 0) or 0)
    reuse_factor = layer.get_attr('reuse_factor') if layer else None
    strategy = layer.get_attr('strategy') if layer else None
    precision = var.type.precision
    layer_class = layer.class_name if layer else None

    key = (backend, io_type, str(strategy).lower(), layer_class, role)
    describe = _ADAPTERS.get(key)
    if describe is None:
        described = {
            'note': (
                f'no adapter for backend={key[0]!r} io_type={key[1]!r} strategy={key[2]!r} '
                f'layer={key[3]!r} role={key[4]!r}; no interface kind, geometry or ordering is claimed -- '
                'classify from the export report. Note that a fully partitioned parameter '
                '(Strategy=Latency, or any ARRAY_PARTITION complete) has no address to bank: '
                'it lowers to one port per element, so it is out of scope by construction '
                'rather than by omission.'
            )
        }
    else:
        described = describe(layer, n_scalars, reuse_factor, precision)

    return ExternalParameter(
        name=var.name,
        layer=layer.name if layer else None,
        layer_class=layer_class,
        role=role,
        tensor_shape=list(getattr(var, 'shape', []) or []),
        n_scalars=n_scalars,
        precision=precision,
        reuse_factor=reuse_factor,
        strategy=str(strategy) if strategy is not None else None,
        **described,
    )


def build_manifest(model):
    """The manifest for a ModelGraph; its parameter list is empty if none apply."""
    config = model.config
    try:
        from hls4ml import __version__ as hls4ml_version
    except ImportError:  # pragma: no cover
        hls4ml_version = None

    parameters = [
        describe_parameter(model, var)
        for var in model.get_weight_variables()
        if str(getattr(var, 'storage', '')).lower() == 'bram'
    ]
    return ExternalParameterManifest(
        config.get_project_name(),
        str(config.get_config_value('Backend')),
        config.get_config_value('IOType'),
        config.get_config_value('Part'),
        config.get_config_value('ClockPeriod'),
        parameters,
        hls4ml_version,
    )


def write_manifest(model):
    """Write the manifest. Returns its path, or None when there is nothing to describe."""
    manifest = build_manifest(model)
    if not manifest.parameters:
        return None
    return manifest.save(model.config.get_output_dir())
