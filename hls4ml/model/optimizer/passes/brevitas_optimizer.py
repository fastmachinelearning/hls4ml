# Inserts quantizer nodes into the model as needed for input/output quantization of layers in brevitas
import numpy as np

from hls4ml.model.optimizer import OptimizerPass

# brevitas rounding_mode -> ap_fixed rounding mode. Anything not listed here is rejected rather than
# silently approximated, since the whole point of these nodes is to reproduce brevitas exactly.
_ROUNDING_MODE_MAP = {
    'ROUND': 'RND_CONV',
    'FLOOR': 'TRN',
    'CEIL': 'RND_INF',
    'ROUND_TO_ZERO': 'TRN_ZERO',
}


def _as_fixed_point_quantizer_attributes(quantization, shape):
    """Translate a brevitas activation quantizer into FixedPointQuantizer attributes.

    Returns None if the quantizer cannot be expressed as a plain fixed-point type, in which case the
    caller should fall back to a QONNX ``Quant`` node.

    ``mask_kbi`` holds the (keep-sign, total bits, integer bits) triple that the bit-exact flow works
    with, broadcast over the tensor shape with a leading batch dimension, as in
    ``hls4ml/converters/pytorch/pquant.py``.
    """
    scale = quantization['scale']
    zeropoint = quantization['zeropoint']
    rounding_mode = quantization['rounding_mode']

    # FixedPointQuantizer is a pure fixed-point type: it has no zero-point, and the scale has to be a
    # power of two for the exponent to be expressible as a number of fractional bits.
    if zeropoint != 0:
        return None
    if scale <= 0:
        return None
    mantissa, exponent = np.frexp(scale)
    if mantissa != 0.5:
        return None
    if rounding_mode not in _ROUNDING_MODE_MAP:
        return None

    if any(s is None for s in shape):
        return None

    bit_width = int(quantization['bit_width'])
    k = int(bool(quantization['signed']))
    # scale == 2 ** (exponent - 1), so the number of fractional bits is 1 - exponent
    frac_bits = 1 - int(exponent)
    integer_bits = bit_width - frac_bits  # includes the sign bit, which is what mask_kbi's I means

    shape = (1,) + tuple(int(s) for s in shape)
    kbi = tuple(np.broadcast_to(np.int16(v), shape) for v in (k, bit_width, integer_bits))

    return {
        'mask_kbi': kbi,
        # brevitas always saturates; narrow_range additionally drops the most negative code
        'SAT': 'SAT_SYM' if quantization['narrow'] else 'SAT',
        'RND': _ROUNDING_MODE_MAP[rounding_mode],
        'fusible': True,  # per-tensor quantization, so the masks are uniform by construction
        'overrides': {},
    }


def _bit_exact_can_handle(model):
    """Whether every layer in the graph is one the bit_exact flow knows how to walk.

    Inserting a FixedPointQuantizer switches the whole model over to the bit_exact flow, which raises
    NotImplementedError on any layer it has no handler for (SimpleRNN and Resize among them). So only
    use the fixed-point representation when the entire graph can be handled; otherwise stay on the
    QONNX Quant path, which does not depend on bit_exact.
    """
    from hls4ml.model.optimizer.passes.bit_exact import _produce_kif

    # These are folded away by the parse_qonnx flow, which runs before bit_exact, so they never
    # reach the bit-exact walk even though they have no handler of their own.
    transient_class_names = ('Quant', 'Constant')

    unhandled = _produce_kif.dispatch(object)
    for node in model.graph.values():
        if node.class_name in transient_class_names:
            continue
        if _produce_kif.dispatch(type(node)) is unhandled:
            return False
    return True


def _as_quant_attributes(quantization):
    """Fall-back representation: a QONNX ``Quant`` node, folded later by the quant_opt passes."""
    return {
        'narrow': quantization['narrow'],
        'rounding_mode': quantization['rounding_mode'],
        'signed': quantization['signed'],
        'bitwidth': quantization['bit_width'],
        'zeropt': quantization['zeropoint'],
        'scale': np.array([quantization['scale']]),
    }


class BrevitasInputOutputOptimizer(OptimizerPass):
    """Takes nodes parsed from brevitas and inserts quantizer nodes into the model if necessary.

    Where the brevitas quantizer maps onto a plain fixed-point type a ``FixedPointQuantizer`` is
    inserted, which lets the model-wide ``bit_exact`` flow derive the precisions and removes the
    double-rounding that a fixed input port would otherwise introduce. Quantizers that need a
    zero-point or a non power-of-two scale fall back to a QONNX ``Quant`` node.
    """

    def match(self, node):
        if ('output_quantization' in node.attributes.keys() and not len(node.attributes['output_quantization']) == 0) or (
            'input_quantization' in node.attributes.keys() and not len(node.attributes['input_quantization']) == 0
        ):
            return True
        else:
            return False

    def _use_fixed_point_quantizer(self, model):
        # An explicit BitExact setting is the user's call; otherwise decide from what the graph contains.
        enabled = model.config.config['HLSConfig']['Model'].get('BitExact', None)
        if enabled is not None:
            return bool(enabled)
        return _bit_exact_can_handle(model)

    def _make_quantizer_node(self, model, quantization, name, inputs, shape):
        attributes = None
        if self._use_fixed_point_quantizer(model):
            attributes = _as_fixed_point_quantizer_attributes(quantization, shape)

        if attributes is not None:
            class_name = 'FixedPointQuantizer'
        else:
            class_name = 'Quant'
            attributes = _as_quant_attributes(quantization)

        quant_node = model.make_node(class_name, name, attributes, inputs)
        quant_node.set_attr('name', name)
        return quant_node

    def transform(self, model, node):
        # See if a quantizer needs to be added for the output
        if 'output_quantization' in node.attributes.keys() and not len(node.attributes['output_quantization']) == 0:
            name = f'quant_output_for_{node.get_attr("name")}'
            quant_node = self._make_quantizer_node(
                model,
                node.attributes['output_quantization'],
                name,
                [node.name],
                node.get_output_variable().shape,
            )
            model.insert_node(quant_node)

            node.attributes['output_quantization'] = {}

        # See if quantizers need to be added for the inputs
        if 'input_quantization' in node.attributes.keys() and not len(node.attributes['input_quantization']) == 0:
            for i, input in enumerate(node.inputs):
                name = f'quant_input_for_{node.get_attr("name")}_input_{i}'
                quant_node = self._make_quantizer_node(
                    model,
                    node.attributes['input_quantization'],
                    name,
                    [input],
                    node.get_input_variable(input).shape,
                )
                model.insert_node(quant_node, input_idx=i)

            node.attributes['input_quantization'] = {}

        return True
