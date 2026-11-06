from warnings import warn

from hls4ml.model.external_parameters import BlockLayout, CompleteLayout, ExternalParameter, FlatOrder, unencodable_reason
from hls4ml.model.optimizer import OptimizerPass

# dense_resource_rf_gt_nin is left out: hls4ml's reuse factor validation never selects it
VERIFIED_DENSE_KERNELS = {'dense_resource_rf_leq_nin', 'dense_resource_rf_gt_nin_rem0'}

# A memory of one word has no address to put a bank id in
MIN_ADDRESSABLE_DEPTH = 2

# Widest reshaped word verified end to end
MAX_RESHAPED_PORT_BITS = 4096


class Unbankable(Exception):
    """The interface HLS builds for a parameter cannot be described."""


def _parameter(layer, role, var, **interface):
    return ExternalParameter(
        name=var.name,
        layer=layer.name,
        layer_class=layer.class_name,
        role=role,
        tensor_shape=list(var.shape),
        n_scalars=var.data_length,
        precision=var.type.precision,
        reuse_factor=layer.get_attr('reuse_factor'),
        strategy=str(layer.get_attr('strategy')),
        **interface,
    )


def _dense_kernel_variant(n_in, reuse_factor):
    """The kernel nnet_dense_resource.h::dense_resource dispatches to."""
    if reuse_factor <= n_in:
        return 'dense_resource_rf_leq_nin'
    if reuse_factor % n_in == 0:
        return 'dense_resource_rf_gt_nin_rem0'
    return 'dense_resource_rf_gt_nin'


def _dense_weight(layer, role, var):
    """``ARRAY_RESHAPE block factor=N``: scalar f lands in word f % depth, lane f // depth."""
    n_in, n_out, reuse_factor = layer.get_attr('n_in'), layer.get_attr('n_out'), layer.get_attr('reuse_factor')
    kernel = _dense_kernel_variant(n_in, reuse_factor)
    if kernel not in VERIFIED_DENSE_KERNELS:
        raise Unbankable(f'Dense kernel {kernel} is not verified')
    lanes = -(-var.data_length // reuse_factor)
    depth = -(-var.data_length // lanes)
    if depth < MIN_ADDRESSABLE_DEPTH:
        raise Unbankable(f'reuse factor {reuse_factor} packs all the values into one word, which cannot be banked')
    width = lanes * var.type.precision.width
    if width > MAX_RESHAPED_PORT_BITS:
        raise Unbankable(f'{width}-bit words are wider than the {MAX_RESHAPED_PORT_BITS} bits verified')
    return _parameter(
        layer,
        role,
        var,
        kernel_variant=kernel,
        interface_kind='bram',
        data_width=width,
        depth=depth,
        flat_order=FlatOrder(['n_in', 'n_out'], ['n_out', 'n_in'], [n_in, n_out]),
        layout=BlockLayout(depth, lanes),
    )


def _scalar_bundle(axis):
    """A fully partitioned bias: one scalar port per element."""

    def describe(layer, role, var):
        return _parameter(
            layer,
            role,
            var,
            interface_kind='scalar_bundle',
            data_width=var.type.precision.width,
            flat_order=FlatOrder([axis], [axis], [var.data_length]),
            layout=CompleteLayout(),
        )

    return describe


def _pointwise_weight(layer, role, var):
    """The pointwise kernel reads one scalar per word, whatever the reuse factor."""
    n_chan, n_filt = layer.get_attr('n_chan'), layer.get_attr('n_filt')
    two_d = layer.get_attr('filt_height') is not None
    if layer.get_attr('filt_width') != 1 or (two_d and layer.get_attr('filt_height') != 1):
        raise Unbankable('the filter is not 1-wide')
    if layer.get_attr('implementation') != 'linebuffer':
        raise Unbankable(f'only the linebuffer implementation is verified, not {layer.get_attr("implementation")!r}')
    if var.data_length < MIN_ADDRESSABLE_DEPTH:
        raise Unbankable('a single value cannot be banked')
    if two_d:
        flat_order = FlatOrder(
            ['filt_height', 'filt_width', 'n_chan', 'n_filt'],
            ['n_filt', 'filt_height', 'filt_width', 'n_chan'],
            [1, 1, n_chan, n_filt],
        )
    else:
        flat_order = FlatOrder(['filt_width', 'n_chan', 'n_filt'], ['n_filt', 'filt_width', 'n_chan'], [1, n_chan, n_filt])
    return _parameter(
        layer,
        role,
        var,
        kernel_variant='pointwise_unreshaped',
        interface_kind='bram',
        data_width=var.type.precision.width,
        depth=var.data_length,
        flat_order=flat_order,
        layout=BlockLayout(var.data_length, 1),
    )


# (io_type, strategy, layer class, role) -> layout. An entry is added only once the layout is verified against RTL.
_LAYOUTS = {
    ('io_parallel', 'resource', 'Dense', 'weight'): _dense_weight,
    ('io_parallel', 'resource', 'Dense', 'bias'): _scalar_bundle('n_out'),
    ('io_parallel', 'resource', 'PointwiseConv1D', 'weight'): _pointwise_weight,
    ('io_parallel', 'resource', 'PointwiseConv1D', 'bias'): _scalar_bundle('n_filt'),
    ('io_parallel', 'resource', 'PointwiseConv2D', 'weight'): _pointwise_weight,
    ('io_parallel', 'resource', 'PointwiseConv2D', 'bias'): _scalar_bundle('n_filt'),
}


def described_combinations():
    return sorted(_LAYOUTS)


def _is_external(var):
    return getattr(var, 'storage', '').lower() == 'bram'


class DescribeExternalParameters(OptimizerPass):
    """Attaches to every external parameter how its values map onto the interface HLS builds for it, and warns
    when a parameter selected with ExternalParameters cannot be placed in parameter banks."""

    def match(self, node):
        return any(_is_external(var) and not hasattr(var, 'external_parameter') for var in node.weights.values())

    def transform(self, model, node):
        selected = node.get_attr('external_parameters') or []
        for role, var in node.weights.items():
            if not _is_external(var) or hasattr(var, 'external_parameter'):
                continue
            key = (model.config.get_config_value('IOType'), str(node.get_attr('strategy')).lower(), node.class_name, role)
            try:
                reason = unencodable_reason(var.type.precision)
                if reason:
                    raise Unbankable(reason)
                if key not in _LAYOUTS:
                    raise Unbankable(f'no verified layout for {key[2]} {role} with io_type={key[0]} and strategy={key[1]}')
                var.external_parameter = _LAYOUTS[key](node, role, var)
            except Unbankable as exc:
                var.external_parameter = _parameter(node, role, var, note=str(exc))
                if role in selected:
                    warn(f'{node.name}: {role} cannot be placed in parameter banks: {exc}', stacklevel=2)
        return False
