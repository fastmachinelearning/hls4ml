from warnings import warn

from hls4ml.model.layers import Layer, Softmax
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType


class MaterializeSoftmaxTables(OptimizerPass):
    """Turn HGQ's trained softmax exp/reciprocal lookup tables into weight arrays.

    Without this the generated C++ rebuilds both tables at runtime from std::exp and 1/x
    (nnet::init_exp_table / nnet::init_invert_table), which does not reproduce HGQ's own
    quantized values. QSoftmaxHandler stashes the builders as _exp_table_fn / _inv_table_fn;
    both take the (k, i, f) of the type the C++ addresses the table with, which is known only
    here: for the latency implementation it is the softmax input precision, decided by
    bit_exact. This pass therefore runs after bit_exact and before transform_types.
    """

    def match(self, node: Layer):
        # Keyed on the stash rather than on the layer type: only the HGQ frontend leaves it,
        # and it must always be removed again, being a callable no saved model can hold.
        return isinstance(node, Softmax) and '_exp_table_fn' in node.attributes

    def transform(self, model, node: Layer):
        exp_table_fn = node.attributes.pop('_exp_table_fn')
        inv_table_fn = node.attributes.pop('_inv_table_fn')

        impl = node.get_attr('implementation')
        if impl not in ('latency', 'stable'):
            return False  # argmax and legacy use no lookup table

        # The exp table is indexed by the normalized input (x_max - x) for the stable
        # implementation and by the layer input itself for the latency one. Reading the
        # element precision off the variable is only unambiguous because this runs before
        # transform_types wraps io_stream variables.
        if impl == 'stable':
            exp_inp_t: FixedPrecisionType = node.attributes['inp_norm_t'].precision
        else:
            exp_inp_t = node.get_input_variable().type.precision
        inv_inp_t: FixedPrecisionType = node.attributes['inv_inp_t'].precision

        for name, addr_t, fn in (('exp_table', exp_inp_t, exp_table_fn), ('inv_table', inv_inp_t, inv_table_fn)):
            # softmax_idx_from_real_val slices the top ceillog2(table_size) bits of the address
            # word. Sizing the table as 2**width makes that the whole word, so no low bits are
            # dropped and the slice cannot run off the end of a narrower word.
            size = 1 << int(addr_t.width)
            if int(node.get_attr(f'{name}_size') or 0) != size:
                warn(
                    f'{node.name}: {name}_size {node.get_attr(f"{name}_size")} does not match the {addr_t.width}-bit '
                    f'address type {addr_t}; overriding to {size} to keep table indexing bit-exact.',
                    stacklevel=1,
                )
            node.set_attr(f'{name}_size', size)
            k = int(bool(addr_t.signed))
            data = fn((k, addr_t.integer - k, int(addr_t.width) - addr_t.integer))

            # Capture the type bit_exact derived first: storing a WeightVariable under `name`
            # makes AttributeDict overwrite `{name}_t` with the variable's own type.
            named_t = node.attributes[f'{name}_t']
            node.set_attr(f'{name}_data', data)
            node.add_weights_variable(
                name=name, var_name=name + '{index}', precision=named_t.precision, type_name=named_t.name
            )
            node.get_weights(name).type = named_t
            node.attributes[f'{name}_t'] = named_t
            model.config.layer_name_precision[f'{node.name}_{name}'] = str(named_t.precision)

        return False
