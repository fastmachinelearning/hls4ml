from copy import copy

import numpy as np

from hls4ml.model.layers import Activation, Dense, HardActivation, ParametrizedActivation, PReLU
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass
from hls4ml.model.types import NamedType

# Activations a Dense kernel can compute on the value it has just produced, grouped by what else the
# kernel needs. softmax is absent by nature: it needs every output of the layer before producing any.

# Computed from the value alone
INLINE_ACTIVATIONS = ('linear', 'relu', 'binary_tanh')

# Computed from the value alone, by reading a table; the kernel also needs the size of the table
TABLE_ACTIVATIONS = ('sigmoid', 'tanh', 'softplus', 'softsign', 'selu')

# Computed from the value and one number shared by every value
SCALAR_PARAM_ACTIVATIONS = ('leaky_relu', 'thresholded_relu', 'elu')

# Computed from the value and two numbers shared by every value
HARD_ACTIVATIONS = ('hard_sigmoid', 'hard_tanh')

FUSED = 'fused'


def reads_interval(node):
    """Whether the reuse factor gives a requested interval rather than its usual meaning.

    Not read with get_layer_config_value: that stops at the first section that exists, so a flag set
    for the model would not be seen once a layer has a section of its own.
    """
    sections = node.model.config.config['HLSConfig']
    value = False
    for section in (
        sections.get('LayerName', {}).get(node.name),
        sections.get('LayerType', {}).get(node.class_name),
        sections.get('Model'),
    ):
        if section is not None and 'ReuseFactorAsInterval' in section:
            value = section['ReuseFactorAsInterval']
            break
    text = str(value).strip().lower()
    if text in ('true', '1'):
        return True
    if text in ('false', '0', 'none', ''):
        return False
    raise Exception(
        f'Layer "{node.name}": ReuseFactorAsInterval is set to {value!r}, which is neither true nor '
        'false. Use true or false (or leave it unset).'
    )


def _is_fused(node):
    return isinstance(node, Dense) and str(node.get_attr('strategy', '')).lower() == FUSED


def _graph_class(node):
    """The class of the layer as the graph defines it.

    A backend makes a subclass of every layer class to add its own attributes, so the class of a layer in
    a built model is VitisActivation rather than Activation and comparing types directly never matches.
    """
    for cls in type(node).__mro__:
        if cls.__module__ == Activation.__module__:
            return cls
    return type(node)


def _foldable_activation(node):
    """Return the activation this layer computes, or None if it cannot be folded.

    The classes are compared exactly rather than with isinstance: they all inherit from Activation, and
    treating a ParametrizedActivation or a PReLU as a plain one would drop the numbers it carries.
    """
    cls = _graph_class(node)

    if cls is Activation:
        name = node.get_attr('activation', '').lower()
        return name if name in INLINE_ACTIVATIONS + TABLE_ACTIVATIONS else None

    if cls is HardActivation:
        name = node.get_attr('activation', '').lower()
        return name if name in HARD_ACTIVATIONS else None

    if cls is ParametrizedActivation:
        name = node._get_act_function_name().lower()
        return name if name in SCALAR_PARAM_ACTIVATIONS else None

    # PReLU is foldable in principle, but its numbers are weights of the activation layer and would
    # have to move to the Dense layer. Left for later.
    if cls is PReLU:
        return None

    return None


class FoldActivationIntoFused(OptimizerPass):
    """Compute an activation at the end of the Dense layer before it and remove the separate layer.

    Besides saving a process in the region, this keeps two Dense layers neighbours, without which
    PlanDenseFusion finds no chain. The numbers the kernel needs are copied to the Dense layer.
    """

    def match(self, node):
        if _foldable_activation(node) is None:
            return False
        prev = node.get_input_node()
        if prev is None or not _is_fused(prev):
            return False
        return len(prev.get_output_nodes()) == 1 and prev.get_attr('fused_activation') is None

    def transform(self, model, node):
        prev = node.get_input_node()
        activation = _foldable_activation(node)
        prev.set_attr('fused_activation', activation)

        # The Dense layer takes over the rounding the activation layer did, so the chain carries the
        # same types as it would without the fold. preact_t is what the activation is computed on.
        out_var = prev.get_output_variable()
        prev.set_attr('fused_preact_t', NamedType(f'{prev.name}_preact_t', copy(out_var.type.precision)))
        out_var.type.precision = copy(node.get_output_variable().type.precision)

        if activation in TABLE_ACTIVATIONS:
            if node.get_attr('table_size') is not None:
                prev.set_attr('fused_table_size', node.get_attr('table_size'))
            # Set on the Dense layer, the type is also declared: the types of a layer are the
            # attributes that hold one.
            if node.get_attr('table_t') is not None:
                prev.set_attr('fused_table_t', node.get_attr('table_t'))

        # Each number keeps the type hls4ml gave it; the three are not the same, and rounding one of
        # them to a different type changes the result.
        if activation in SCALAR_PARAM_ACTIVATIONS:
            prev.set_attr('fused_activation_param', node.get_attr('activ_param', 1.0))
            if node.get_attr('param_t') is not None:
                prev.set_attr('fused_param_t', node.get_attr('param_t'))

        if activation in HARD_ACTIVATIONS:
            prev.set_attr('fused_activation_slope', node.get_attr('slope', 0.2))
            prev.set_attr('fused_activation_shift', node.get_attr('shift', 0.5))
            if node.get_attr('slope_t') is not None:
                prev.set_attr('fused_slope_t', node.get_attr('slope_t'))
            if node.get_attr('shift_t') is not None:
                prev.set_attr('fused_shift_t', node.get_attr('shift_t'))

        model.remove_node(node)
        return True


class PlanDenseFusion(ModelOptimizerPass):
    """Group consecutive Dense layers into regions and select the kernel each one uses.

    The selected kernel is stored on the layer as the attribute ``fused_form``, and this pass sets
    three attributes in total:

    * ``fused_form``        - which of the three kernels computes the layer: dot, axpy or plain
    * ``fused_multipliers`` - how many multipliers the kernel uses at the same time
    * ``fused_stream_out``  - whether the output is written one value at a time, as a stream

    There are three kernels because a Dense layer needs all of its inputs before it can produce any
    output, so it can pass data one value at a time on one side only. ``dot`` reads an array and
    writes one value at a time; ``axpy`` reads one value at a time and writes an array; ``plain``
    reads and writes arrays. The layers of a region are given dot, axpy, dot, axpy and so on, so that
    each dot and the axpy after it can run at the same time, one consuming what the other produces.
    A region with an odd number of layers starts with a plain layer, so that the region as a whole
    still begins and ends with an array, which is what the rest of the model expects.
    """

    def __init__(self):
        pass

    def transform(self, model):
        # One read of an io_stream connection carries a whole row, which these kernels cannot use.
        # The validation pass reports it; here it only stops the pass.
        if model.config.get_config_value('IOType') != 'io_parallel':
            return False

        changed = False
        fused_layers = []
        for run in self._dense_runs(model):
            forms = self._assign_forms(len(run))
            for layer, form in zip(run, forms):
                if layer.get_attr('fused_form') != form:
                    layer.set_attr('fused_form', form)
                    changed = True
                fused_layers.append(layer)

        if not changed:
            return False

        self._set_parallel_multipliers(fused_layers)
        self._mark_streamed_outputs(fused_layers)

        # Layers of a region run at the same time, which needs DATAFLOW rather than a pipeline
        if any(layer.get_attr('fused_form') in ('dot', 'axpy') for layer in fused_layers):
            model.config.pipeline_style = 'dataflow'

        return True

    def _assign_forms(self, length):
        """Return the form of each layer of a chain of the given length."""
        if length < 2:
            return ['plain'] * length
        head = ['plain'] if length % 2 else []
        body = ['dot' if k % 2 == 0 else 'axpy' for k in range(length - len(head))]
        return head + body

    def _dense_runs(self, model):
        """Return the chains of Dense layers that can be fused.

        A chain runs while each layer uses the fused strategy and is the only reader of the one before
        it. Another layer in between, or a second reader, ends it.
        """
        runs, current = [], []
        for layer in model.get_layers():
            if _is_fused(layer) and len(layer.get_output_nodes()) <= 1:
                if current and layer.get_input_node() is not current[-1]:
                    runs.append(current)
                    current = []
                current.append(layer)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return [run for run in runs if len(run) > 1]

    def _most_usable_multipliers(self, layer):
        """Multipliers beyond the dimension the kernel iterates over would be unused."""
        if layer.get_attr('fused_form') == 'dot':
            return int(layer.get_attr('n_in'))
        return int(layer.get_attr('n_out'))

    def _reads_interval(self, layer):
        return reads_interval(layer)

    def _work_cycles(self, layer, multipliers):
        """Cycles of computation for one run of the layer, excluding the wait states."""
        n_in, n_out = int(layer.get_attr('n_in')), int(layer.get_attr('n_out'))
        trips = -(-self._most_usable_multipliers(layer) // multipliers)
        if layer.get_attr('fused_form') == 'dot':
            return n_out * trips
        # Every input, plus the pass that applies the activation
        return (n_in + 1) * trips

    def _headroom_cycles(self, layer, multipliers):
        """Cycles to fill the pipeline and pass data between layers. An upper estimate, since the
        exact value depends on the layer and the tool version, so the interval is never too large."""
        return 10 + -(-self._most_usable_multipliers(layer) // multipliers)

    def _set_from_interval(self, layer):
        """With ReuseFactorAsInterval the reuse factor is the largest interval the layer may have.

        Use the fewest multipliers that stay within it and spend what remains as wait states. If no
        number of multipliers is enough, record the smallest interval the layer can have instead, for
        the validation pass to report.
        """
        target = max(1, int(layer.get_attr('reuse_factor', 1) or 1))
        cap = self._most_usable_multipliers(layer)
        for multipliers in range(1, cap + 1):
            predicted = self._work_cycles(layer, multipliers) + self._headroom_cycles(layer, multipliers)
            if predicted <= target:
                layer.set_attr('fused_multipliers', multipliers)
                layer.set_attr('fused_pad_cycles', target - predicted)
                # How much smaller the interval may be, since the headroom is an upper estimate
                layer.set_attr('fused_interval_slack', self._headroom_cycles(layer, multipliers) - 8)
                return
        floor = self._work_cycles(layer, cap) + self._headroom_cycles(layer, cap)
        layer.set_attr('fused_multipliers', cap)
        layer.set_attr('fused_pad_cycles', 0)
        layer.set_attr('fused_interval_floor', floor)

    def _set_parallel_multipliers(self, layers):
        """Derive the multiplier count from the reuse factor, as the other strategies do, so the same
        reuse factor asks for the same hardware here as it does there."""
        for layer in layers:
            if self._reads_interval(layer):
                self._set_from_interval(layer)
                continue
            n_in = int(layer.get_attr('n_in'))
            n_out = int(layer.get_attr('n_out'))
            reuse = max(1, int(layer.get_attr('reuse_factor', 1) or 1))
            wanted = max(1, (n_in * n_out) // reuse)
            # More multipliers than values to work through would leave some unused
            layer.set_attr('fused_multipliers', min(wanted, self._most_usable_multipliers(layer)))

        # A pair runs only as fast as its slower half, so the faster half cannot use its extra
        # multipliers. Skip interval-configured layers: they already have the fewest that fit.
        for layer in layers:
            if layer.get_attr('fused_form') != 'dot' or self._reads_interval(layer):
                continue
            consumers = layer.get_output_nodes()
            if consumers and consumers[0].get_attr('fused_form') == 'axpy' and not self._reads_interval(consumers[0]):
                pair = (layer, consumers[0])
                shared = min(int(n.get_attr('fused_multipliers')) for n in pair)
                for n in pair:
                    n.set_attr('fused_multipliers', shared)

    def _mark_streamed_outputs(self, layers):
        """Mark the outputs a later pass, TransformTypes, turns into streams: only a dot layer read
        by an axpy layer writes one value at a time."""
        for layer in layers:
            consumers = layer.get_output_nodes()
            streams = (
                layer.get_attr('fused_form') == 'dot'
                and len(consumers) == 1
                and consumers[0].get_attr('fused_form') == 'axpy'
            )
            layer.set_attr('fused_stream_out', bool(streams))


class LayoutFusedDotWeights(OptimizerPass):
    """Transpose the weights of a dot layer into the order that kernel reads them.

    hls4ml stores a Dense weight for input i and output j at i * n_out + j, which is what axpy reads.
    Every other form produces one output at a time and needs the inputs of one output together, at
    j * n_in + i. That includes a layer the planner gave no form, which is a layer the strategy was
    asked for that is not part of a chain: it is computed by the same kernel as a leading layer.
    """

    def match(self, node):
        return _is_fused(node) and node.get_attr('fused_form') != 'axpy' and not node.get_attr('fused_weights_transposed')

    def transform(self, model, node):
        weight = node.weights['weight']
        weight.data = np.ascontiguousarray(weight.data.T)
        weight.shape = list(weight.data.shape)
        node.set_attr('fused_weights_transposed', True)
        return True
