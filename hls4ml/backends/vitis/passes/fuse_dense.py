import math
from copy import copy

import numpy as np

from hls4ml.model.layers import (
    Activation,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    HardActivation,
    ParametrizedActivation,
)
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass
from hls4ml.model.types import NamedType

# Activations the fused kernels can apply to each output as it is computed, grouped by what else the
# kernel needs. Softmax cannot be one of them: it needs every output of the layer before it can produce any.

# Computed from the value alone
INLINE_ACTIVATIONS = ('linear', 'relu', 'binary_tanh')

# Read from a lookup table; the kernel also needs the table size
TABLE_ACTIVATIONS = ('sigmoid', 'tanh', 'softplus', 'softsign', 'selu')

# Need one parameter, such as the slope of leaky_relu
SCALAR_PARAM_ACTIVATIONS = ('leaky_relu', 'thresholded_relu', 'elu')

# Need two parameters, a slope and a shift
HARD_ACTIVATIONS = ('hard_sigmoid', 'hard_tanh')

FUSED = 'fused'


def _reads_interval(node):
    """Whether ReuseFactorAsInterval is set for this layer.

    Not read with get_layer_config_value, which looks only in the first section it finds, and so misses
    a value set for the model when the layer has a section of its own.
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
    return _read_flag(value, f'Layer "{node.name}"', 'ReuseFactorAsInterval')


def _read_flag(value, where, key):
    """Read a true/false setting, which is a boolean in Python and may be a string in JSON or YAML."""

    text = str(value).strip().lower()
    if text in ('true', '1'):
        return True
    if text in ('false', '0', 'none', ''):
        return False
    raise Exception(
        f'{where}: {key} is set to {value!r}, which is neither true nor false. Use true or false (or leave it unset).'
    )


def _readers(layer):
    """How many times the output of the layer is read, counting a model output as one more reader."""
    return len(layer.get_output_nodes()) + any(name in layer.model.outputs for name in layer.outputs)


def _find_chains(model, in_chain, between=None):
    """Return lists of layers for which `in_chain` is true, each reading the one before it.

    One layer for which `between` is true may sit between two layers of a chain without ending it. Each
    chain is followed from a layer to the layer that reads it, so chains on parallel branches are found.
    """

    def next_in_chain(layer):
        readers = layer.get_output_nodes()
        if len(readers) != 1:
            return None
        if in_chain(readers[0]):
            return readers[0]
        if between is not None and between(readers[0]):
            after = readers[0].get_output_nodes()
            if len(after) == 1 and in_chain(after[0]):
                return after[0]
        return None

    chains, seen = [], set()
    # In graph order, so that each chain is reached through its first layer
    for layer in model.get_layers():
        if layer.name in seen or not in_chain(layer):
            continue
        chain = [layer]
        while (following := next_in_chain(chain[-1])) is not None:
            chain.append(following)
        seen.update(node.name for node in chain)
        chains.append(chain)
    return chains


def _is_fused(node):
    return isinstance(node, Dense) and str(node.get_attr('strategy', '')).lower() == FUSED


def _graph_class(node):
    """Return the hls4ml class of the layer, such as Activation.

    Each backend subclasses the layer classes (VitisActivation, for example), so comparing the type of a
    layer directly never matches.
    """
    for cls in type(node).__mro__:
        if cls.__module__ == Activation.__module__:
            return cls
    return type(node)


def _foldable_activation(node):
    """Return the name of the activation this layer computes, or None if it cannot be folded.

    Classes are compared exactly, not with isinstance: they all inherit from Activation, and treating a
    ParametrizedActivation or a PReLU as a plain one would lose its parameters.
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

    # PReLU could be folded too, but its parameters are stored as weights of the activation layer and
    # would have to be moved to the Dense layer. Not done yet.
    return None


class FoldActivationIntoFused(OptimizerPass):
    """Move an activation into the fused Dense layer before it and remove the activation layer.

    Only done for Dense layers the planner put in a chain; any other layer keeps its activation layer,
    since its kernel cannot compute the activation. The parameters of the activation are copied to the
    Dense layer.
    """

    def match(self, node):
        if _foldable_activation(node) is None:
            return False
        prev = node.get_input_node()
        if prev is None or prev.get_attr('fused_form') is None:
            return False
        return len(prev.get_output_nodes()) == 1 and prev.get_attr('fused_activation') is None

    def transform(self, model, node):
        prev = node.get_input_node()
        activation = _foldable_activation(node)
        prev.set_attr('fused_activation', activation)

        # The Dense layer takes the output type of the activation layer, so the values passed along the
        # chain keep their types. preact_t, the type the activation is applied to, is the old output type.
        out_var = prev.get_output_variable()
        prev.set_attr('fused_preact_t', NamedType(f'{prev.name}_preact_t', copy(out_var.type.precision)))
        out_var.type.precision = copy(node.get_output_variable().type.precision)

        if activation in TABLE_ACTIVATIONS or activation == 'elu':
            if node.get_attr('table_size') is not None:
                prev.set_attr('fused_table_size', node.get_attr('table_size'))
            # Storing the type as an attribute of the Dense layer is enough for it to be declared
            if node.get_attr('table_t') is not None:
                prev.set_attr('fused_table_t', node.get_attr('table_t'))

        # Each parameter keeps the type it had in the activation layer. The types differ, and using
        # another one changes the result.
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

    It sets three attributes on each layer of a chain:

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
        changed = False
        chains = self._dense_chains(model)
        for chain in chains:
            for layer, form in zip(chain, self._assign_forms(len(chain))):
                if layer.get_attr('fused_form') != form:
                    layer.set_attr('fused_form', form)
                    changed = True

        if not changed:
            return False

        self._set_parallel_multipliers(chains)
        self._mark_streamed_outputs(chains)

        # Layers of a region run at the same time, which needs DATAFLOW rather than a pipeline
        if any(len(chain) > 1 for chain in chains):
            model.config.pipeline_style = 'dataflow'

        return True

    def _assign_forms(self, length):
        """Return the form of each layer of a chain of the given length."""
        if length < 2:
            return ['plain'] * length
        head = ['plain'] if length % 2 else []
        body = ['dot' if k % 2 == 0 else 'axpy' for k in range(length - len(head))]
        return head + body

    def _dense_chains(self, model):
        """Return the chains of two or more Dense layers that can be fused.

        Each layer of a chain uses the fused strategy, reads the layer before it, and has its output read
        by at most one layer, where a model output counts as a reader. An activation that
        FoldActivationIntoFused can move into the Dense layer may sit between two of them.
        """

        def in_chain(layer):
            return _is_fused(layer) and _readers(layer) <= 1

        def between(layer):
            return _foldable_activation(layer) is not None and _readers(layer) <= 1

        return [chain for chain in _find_chains(model, in_chain, between) if len(chain) > 1]

    def _most_usable_multipliers(self, layer):
        """The most multipliers the kernel can use: one per value of the dimension it works through."""
        if layer.get_attr('fused_form') == 'dot':
            return int(layer.get_attr('n_in'))
        return int(layer.get_attr('n_out'))

    def _work_cycles(self, layer, multipliers):
        """Cycles the layer spends computing one input, excluding the wait states."""
        n_in, n_out = int(layer.get_attr('n_in')), int(layer.get_attr('n_out'))
        trips = -(-self._most_usable_multipliers(layer) // multipliers)
        if layer.get_attr('fused_form') == 'dot':
            return n_out * trips
        # Every input, plus the pass that adds the bias and applies the activation
        return (n_in + 1) * trips

    def _headroom_cycles(self, layer, multipliers):
        """Cycles beyond the work: the depth of the kernel's pipelines, which are measured upper bounds, and 4
        for passing data between the layers of a region."""
        form = layer.get_attr('fused_form')
        trips = -(-self._most_usable_multipliers(layer) // multipliers)
        if form == 'dot':
            depth = 10
        elif trips == 1:
            depth = 5
        elif form == 'axpy':
            depth = 16
        else:
            # Selecting one of n_in inputs takes a pipeline stage per doubling of n_in
            depth = 16 + math.ceil(math.log2(int(layer.get_attr('n_in'))))
        return depth + 4

    def _cycles(self, layer, multipliers):
        """The planner's estimate of the interval of the layer with this many multipliers."""
        return self._work_cycles(layer, multipliers) + self._headroom_cycles(layer, multipliers)

    def _set_from_interval(self, layer):
        """Use the fewest multipliers that keep the layer within the requested interval, and fill the
        remaining cycles with wait states. If even all of them are not enough, record the smallest
        interval the layer can reach, which ValidateDenseFusion reports as an error.

        Only numbers of multipliers that divide the dimension the kernel works through are considered, since
        any other number makes the pipelines deeper than _headroom_cycles allows for.
        """
        target = max(1, int(layer.get_attr('reuse_factor', 1) or 1))
        cap = self._most_usable_multipliers(layer)
        for multipliers in (m for m in range(1, cap + 1) if cap % m == 0):
            predicted = self._cycles(layer, multipliers)
            if predicted <= target:
                layer.set_attr('fused_multipliers', multipliers)
                layer.set_attr('fused_pad_cycles', target - predicted)
                # How much smaller than requested the interval may come out, 3 being the smallest
                # overhead measured
                layer.set_attr('fused_interval_slack', self._headroom_cycles(layer, multipliers) - 3)
                return
        floor = self._cycles(layer, cap)
        layer.set_attr('fused_multipliers', cap)
        layer.set_attr('fused_pad_cycles', 0)
        layer.set_attr('fused_interval_floor', floor)

    def _set_parallel_multipliers(self, chains):
        """Set the multiplier count from the reuse factor the way the other strategies do, so that a
        reuse factor asks for the same hardware under every strategy, then lower it where more would not
        make the chain faster."""
        for layer in (layer for chain in chains for layer in chain):
            if _reads_interval(layer):
                self._set_from_interval(layer)
                continue
            n_in = int(layer.get_attr('n_in'))
            n_out = int(layer.get_attr('n_out'))
            reuse = max(1, int(layer.get_attr('reuse_factor', 1) or 1))
            wanted = max(1, (n_in * n_out) // reuse)
            layer.set_attr('fused_multipliers', min(wanted, self._most_usable_multipliers(layer)))

        # A chain mixing the two readings of the reuse factor is an error, reported by ValidateDenseFusion
        for chain in chains:
            if all(_reads_interval(layer) for layer in chain):
                self._share_interval(chain)
            elif not any(_reads_interval(layer) for layer in chain):
                self._balance(chain)

    def _balance(self, chain):
        """A chain runs only as fast as its slowest layer. Give every layer the fewest multipliers that
        divide its width and keep it within the cycles of the slowest one, never more than its reuse
        factor allows."""
        allowed = {layer.name: int(layer.get_attr('fused_multipliers')) for layer in chain}
        slowest = max(chain, key=lambda layer: self._cycles(layer, allowed[layer.name]))
        limit = self._cycles(slowest, allowed[slowest.name])
        for layer in chain:
            width = self._most_usable_multipliers(layer)
            fits = [m for m in range(1, allowed[layer.name]) if width % m == 0 and self._cycles(layer, m) <= limit]
            if fits:
                layer.set_attr('fused_multipliers', min(fits))
                layer.set_attr('fused_multipliers_allowed', allowed[layer.name])
                layer.set_attr('fused_slowest_layer', slowest.name)
                layer.set_attr('fused_slowest_cycles', limit)

    def _share_interval(self, chain):
        """Set what the layers of a chain with ReuseFactorAsInterval must agree on, since the chain has a
        single interval: the smallest interval it can reach, and the wait of each dot and axpy pair."""
        if any(layer.get_attr('fused_interval_floor') is not None for layer in chain):
            # The smallest interval the chain can reach, which every layer must then request
            floor = max(self._cycles(layer, self._most_usable_multipliers(layer)) for layer in chain)
            for layer in chain:
                layer.set_attr('fused_region_floor', floor)

        # The faster layer of a pair waits for the slower one before its wait states begin, so both wait
        # only for the cycles the slower one leaves
        for dot, axpy in self._pairs([chain]):
            pad = min(int(n.get_attr('fused_pad_cycles') or 0) for n in (dot, axpy))
            for n in (dot, axpy):
                n.set_attr('fused_pad_cycles', pad)

    def _mark_streamed_outputs(self, chains):
        """Mark the outputs TransformTypes later turns into streams: those of dot layers read by an axpy
        layer, which are written one value at a time."""
        streamed = {dot for dot, _ in self._pairs(chains)}
        for layer in (layer for chain in chains for layer in chain):
            layer.set_attr('fused_stream_out', layer in streamed)

    @staticmethod
    def _pairs(chains):
        """Each dot layer with the axpy layer after it in the same chain. An activation may still sit
        between the two at this point; FoldActivationIntoFused removes it later."""
        for chain in chains:
            for first, second in zip(chain, chain[1:]):
                if first.get_attr('fused_form') == 'dot' and second.get_attr('fused_form') == 'axpy':
                    yield first, second


class SubstituteUnfusedStrategy(OptimizerPass):
    """Switch layers that asked for the fused strategy but could not be fused to the resource strategy.

    These are the layers the planner did not put in a chain: a Dense layer on its own, a Conv1D or a
    Conv2D. Depthwise convolutions keep the latency strategy.
    """

    def match(self, node):
        return (
            isinstance(node, (Dense, Conv1D, Conv2D))
            and not isinstance(node, (DepthwiseConv1D, DepthwiseConv2D))
            and str(node.model.config.get_strategy(node)).lower() == FUSED
            and node.get_attr('fused_form') is None
            and str(node.get_attr('strategy', '')).lower() != 'resource'
        )

    def transform(self, model, node):
        backend = model.config.backend
        n_in, n_out = backend.get_layer_mult_size(node)
        backend.set_target_reuse_factor(node)
        backend.set_closest_reuse_factor(node, n_in, n_out)
        node.set_attr('strategy', 'resource')
        # hls4ml sets dataflow for a model with a resource layer, but the pass that does it reads the
        # configuration, which still says fused. A style the user chose is left alone.
        if model.config.pipeline_style in (None, 'auto'):
            model.config.pipeline_style = 'dataflow'
        return False


class LayoutFusedDotWeights(OptimizerPass):
    """Transpose the weights of dot layers into the order their kernel reads them.

    hls4ml stores the weight for input i and output j at i * n_out + j, which is what the axpy and plain
    kernels read: all outputs for one input at a time. The dot kernel reads one output at a time, at
    j * n_in + i.
    """

    def match(self, node):
        return _is_fused(node) and node.get_attr('fused_form') == 'dot' and not node.get_attr('fused_weights_transposed')

    def transform(self, model, node):
        weight = node.weights['weight']
        weight.data = np.ascontiguousarray(weight.data.T)
        weight.shape = list(weight.data.shape)
        node.set_attr('fused_weights_transposed', True)
        return True


class ValidateDenseFusion(ModelOptimizerPass):
    """Check what the fusion passes decided, then report what was built.

    For each Dense layer that asked for the strategy it checks the reuse factor (an error for 1, a
    warning for a value the layer cannot reach) or, with ReuseFactorAsInterval, that the requested
    interval can be reached and is the same for every layer of a region.

    The report comes after all checks, so nothing is reported for a conversion that stops with an error.
    It lists each chain, each layer that asked for the strategy but was not fused, and the pipeline style
    if the fusion passes changed it. FusedReport: false in the Model section turns it off; warnings and
    errors are always printed. The backend and io_parallel are checked earlier, by ValidateFusedStrategy
    and ValidateFusedIoType.
    """

    def __init__(self):
        pass

    def transform(self, model):
        asked = [node for node in model.get_layers() if self._asked_for_fusion(node)]
        if not asked:
            return False
        for node in asked:
            if not isinstance(node, Dense):
                continue
            if _reads_interval(node) and node.get_attr('fused_form') is not None:
                self._check_interval(node)
            else:
                self._check_reuse_factor(node)
        self._report(model, asked)
        return False

    @staticmethod
    def _asked_for_fusion(node):
        return node.get_attr('strategy') is not None and str(node.model.config.get_strategy(node)).lower() == 'fused'

    def _check_interval(self, node):
        """Check a layer that uses ReuseFactorAsInterval, and print what was built for it."""

        target = max(1, int(node.get_attr('reuse_factor', 1) or 1))

        floor = node.get_attr('fused_interval_floor')
        if floor is not None:
            region = node.get_attr('fused_region_floor') or floor
            needs = f'it still needs {floor} cycles'
            if region > floor:
                needs += f', and the region it is fused into needs {region}'
            raise Exception(
                f'Layer "{node.name}" ({node.class_name}) cannot achieve an interval of {target}. With '
                f'all of its multipliers in use {needs}. The layers of a region request the same interval, '
                f'so request {region} or more for each of them, or leave ReuseFactorAsInterval unset so the '
                'reuse factor keeps its usual meaning.'
            )

        producer = node.get_input_node()
        if producer is not None and producer.get_attr('fused_form') is not None and not _reads_interval(producer):
            raise Exception(
                f'Layers "{producer.name}" and "{node.name}" are fused into one region, but only the '
                'second uses its reuse factor as an interval. A region has a single interval, so its '
                'layers must use the reuse factor the same way.'
            )

        consumers = node.get_output_nodes()
        neighbour = consumers[0] if consumers else None
        if neighbour is not None and neighbour.get_attr('fused_form') is not None:
            if not _reads_interval(neighbour):
                raise Exception(
                    f'Layers "{node.name}" and "{neighbour.name}" are fused into one region, but only '
                    'the first uses its reuse factor as an interval. A region has a single interval, so '
                    'its layers must use the reuse factor the same way.'
                )
            neighbour_target = max(1, int(neighbour.get_attr('reuse_factor', 1) or 1))
            if neighbour_target != target:
                raise Exception(
                    f'Layers "{node.name}" and "{neighbour.name}" are fused into one region but request '
                    f'intervals of {target} and {neighbour_target}. A region has a single interval, so '
                    'its layers must request the same one.'
                )

        built = node.get_attr('fused_multipliers')
        pad = node.get_attr('fused_pad_cycles') or 0
        slack = node.get_attr('fused_interval_slack') or 0
        result = f'between {target - slack} and {target}' if slack else f'{target}'
        print(
            f'Layer "{node.name}": reuse factor {target} used as a requested interval. Built with '
            f'{built} multipliers and {pad} wait cycles. The interval will be {result} cycles.'
        )

    def _check_reuse_factor(self, node):
        """Check the reuse factor in two stages: reject 1, then report a value the layer cannot reach.

        Reuse factor 1 asks for a fully parallel layer, which these kernels cannot build.
        """

        asked = max(1, int(node.get_attr('reuse_factor', 1) or 1))
        built = node.get_attr('fused_multipliers_allowed') or node.get_attr('fused_multipliers')
        # The smallest reuse factor that still makes a difference: below it the layer already uses all
        # the multipliers it can. A layer outside a chain has neither number.
        lowest_usable = None
        if built is not None:
            built = int(built)
            lowest_usable = int(node.get_attr('n_in')) * int(node.get_attr('n_out')) // built

        if asked == 1:
            advice = f'For the most parallel fused design set ReuseFactor to {lowest_usable}. ' if lowest_usable else ''
            raise Exception(
                f'Layer "{node.name}" ({node.class_name}) has strategy "fused" with reuse factor 1. The '
                'fused strategy shares multipliers over several cycles and cannot build a fully parallel '
                f'layer. {advice}For a fully parallel layer use strategy "Latency", or '
                '"distributed_arithmetic".'
            )

        if lowest_usable is None or lowest_usable <= asked:
            return
        print(
            f'WARNING: Layer "{node.name}" ({node.class_name}) asks for reuse factor {asked} with '
            f'strategy "fused", which cannot be built: the {node.get_attr("fused_form")} form uses at '
            f'most {built} multipliers at a time, which is reuse factor {lowest_usable}. The layer is built '
            f'with {node.get_attr("fused_multipliers")}.'
        )

    def _report(self, model, asked):
        """State what was built: each chain, and each layer that asked for the strategy and was not fused.
        FusedReport turns off the first; the second is a warning and is always printed."""

        section = model.config.config['HLSConfig'].get('Model') or {}
        reported = True
        if 'FusedReport' in section:
            reported = _read_flag(section['FusedReport'], 'The Model section', 'FusedReport')
        self._report_style(model, section, reported)

        if reported:
            for chain in _find_chains(model, lambda layer: layer.get_attr('fused_form') is not None):
                layers = ', '.join(
                    f'{layer.name} ({layer.get_attr("fused_form")}, {layer.get_attr("fused_multipliers")} multipliers)'
                    for layer in chain
                )
                print(f'Fused strategy: {layers} are computed as one region.')
                for layer in chain:
                    if layer.get_attr('fused_slowest_layer') is not None:
                        print(
                            f'Fused strategy: {layer.name} uses {layer.get_attr("fused_multipliers")} of the '
                            f'{layer.get_attr("fused_multipliers_allowed")} multipliers its reuse factor allows, '
                            'since more would not make its region faster: '
                            f'{layer.get_attr("fused_slowest_layer")} needs about '
                            f'{layer.get_attr("fused_slowest_cycles")} cycles.'
                        )

        for layer in asked:
            if layer.get_attr('fused_form') is not None:
                continue
            reason = (
                'which needs two or more Dense layers in sequence, each read only by the layer after it'
                if isinstance(layer, Dense)
                else 'which is implemented for Dense layers only'
            )
            print(
                f'WARNING: Layer "{layer.name}" ({layer.class_name}) asked for strategy "fused", '
                f'{reason}. It is built with strategy "{layer.get_attr("strategy")}" and reuse '
                f'factor {layer.get_attr("reuse_factor")}.'
            )

    @staticmethod
    def _report_style(model, section, reported):
        """Report a pipeline style the fusion passes set. Replacing a style from the configuration is a
        warning, printed even when the report is turned off."""

        configured = str(section.get('PipelineStyle', 'auto')).lower()
        style = str(model.config.pipeline_style).lower()
        if style == configured:
            return
        if configured not in ('auto', 'none'):
            print(f'WARNING: PipelineStyle "{configured}" replaced with "{style}".')
        elif reported:
            print(f'Fused strategy: the model is built with pipeline style "{style}".')
