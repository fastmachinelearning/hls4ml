"""Infer the storage layout of every tensor from the operations that produce and consume it.

Each tensor carries a permutation ``perm`` from the axis order of the source framework (batch axis excluded) to the
order in which hls4ml stores it: stored axis ``i`` holds source axis ``perm[i]``. Operations that work on the channel
axis of a channels-first tensor (convolutions, pooling, ...) need it stored channels-last; operations that work on the
last axis (Dense, LayerNorm, RNNs, ...) need it stored in source order; element-wise operations accept any order. A
Transpose is inserted wherever a tensor is not stored the way its consumer needs.

The layouts at the model inputs and outputs are set by ``InputLayout`` and ``OutputLayout`` in the model
configuration: ``'source'`` keeps the source framework's order, ``'native'`` uses the order the adjacent operation
needs. Either can be a dict keyed by tensor name. The resulting layouts are stored in ``model.io_layouts``.

Similar in spirit to https://github.com/fastmachinelearning/qonnx/blob/main/src/qonnx/transformation/channels_last.py
"""

import math
import warnings

import numpy as np

from hls4ml.model.layers import (
    GRU,
    LSTM,
    Activation,
    ApplyAlpha,
    BatchNormalization,
    Concatenate,
    Constant,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Cropping1D,
    Cropping2D,
    Dense,
    Einsum,
    EinsumDense,
    GlobalPooling1D,
    GlobalPooling2D,
    IFNeuron,
    Input,
    LayerNormalization,
    LIFNeuron,
    Merge,
    Pooling1D,
    Pooling2D,
    PReLU,
    Reshape,
    Resize,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    SNNReadout,
    Softmax,
    StreamingSNN,
    Transpose,
    ZeroPadding1D,
    ZeroPadding2D,
)
from hls4ml.model.optimizer import ModelOptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer

_channel_axis_layers = (
    Conv1D,
    Conv2D,
    Conv1DTranspose,
    Conv2DTranspose,
    SeparableConv1D,
    SeparableConv2D,
    Pooling1D,
    Pooling2D,
    GlobalPooling1D,
    GlobalPooling2D,
    BatchNormalization,
    Resize,
    ZeroPadding1D,
    ZeroPadding2D,
    Cropping1D,
    Cropping2D,
)
_last_axis_layers = (
    Dense,
    LayerNormalization,
    SimpleRNN,
    LSTM,
    GRU,
    Einsum,
    EinsumDense,
    Reshape,
    Constant,
    IFNeuron,
    LIFNeuron,
    SNNReadout,
    StreamingSNN,
)
_elementwise_layers = (Activation, Merge, ApplyAlpha, FixedPointQuantizer)


def _identity(rank):
    return tuple(range(rank))


def _channels_last(rank):
    return tuple(range(1, rank)) + (0,)


def transpose_perm(state, target):
    """The ``perm`` of a Transpose that turns a tensor stored as ``state`` into one stored as ``target``."""
    return [state.index(t) for t in target]


def _layout_kind(node):
    if isinstance(node, ApplyAlpha):  # A BatchNormalization subclass, but element-wise
        return 'elementwise'
    if isinstance(node, _channel_axis_layers):
        return 'channels_last'
    if isinstance(node, Transpose):
        return 'transpose'
    if isinstance(node, Concatenate):
        return 'concat'
    if isinstance(node, Softmax):
        return 'softmax'
    if isinstance(node, PReLU):
        return 'elementwise' if np.size(node.get_attr('param_data')) == 1 else 'source'
    if isinstance(node, _elementwise_layers):
        return 'elementwise'
    if isinstance(node, _last_axis_layers):
        return 'source'
    return None


def _model_config(model):
    return model.config.config.get('HLSConfig', {}).get('Model', {})


def _contract(model, key, name):
    value = _model_config(model).get(key, 'source')
    if isinstance(value, dict):
        value = value.get(name, 'source')
    if value not in ('source', 'native'):
        raise ValueError(f"{key} for '{name}' must be 'source' or 'native', got '{value}'.")
    return value


class LayoutInference(ModelOptimizerPass):
    """Stores every tensor in the order its consumers need, inserting Transpose layers where they disagree."""

    def __init__(self):
        self.name = 'layout_inference'

    def transform(self, model):
        if not _model_config(model).get('LayoutInference', False) or getattr(model, 'io_layouts', None) is not None:
            return False

        self.model = model
        self.states = {}
        self.transposes = {}
        self.warned_classes = set()
        self.nop_transposes = []

        io_layouts = {'inputs': [], 'outputs': []}
        for node in list(model.graph.values()):
            if isinstance(node, Input):
                io_layouts['inputs'].append(self._set_input_layout(node))
            else:
                self._process(node)

        for i, out_name in enumerate(list(model.outputs)):
            io_layouts['outputs'].append(self._set_output_layout(i, out_name))

        for node in self.nop_transposes:
            model.remove_node(node)
        for entry, name in zip(io_layouts['outputs'], model.outputs):
            entry['name'] = name

        model.io_layouts = io_layouts
        for kind in ('inputs', 'outputs'):
            for entry in io_layouts[kind]:
                if entry['layout'] == 'native' and entry['perm'] != sorted(entry['perm']):
                    print(
                        f"Layout of {kind[:-1]} '{entry['name']}': source shape {entry['source_shape']} is stored as "
                        f'{[entry["source_shape"][i] for i in entry["perm"]]} (perm {entry["perm"]}).'
                    )
        return True

    def _set_input_layout(self, node):
        var = node.get_output_variable()
        source_shape = list(var.shape)
        rank = len(source_shape)
        layout = _contract(self.model, 'InputLayout', node.name)
        perm = _identity(rank)
        if layout == 'native' and rank > 1:
            perm = self._preferred_layout(node.outputs[0], rank)
        var.shape = [source_shape[i] for i in perm]
        self.states[node.outputs[0]] = perm
        return {'name': node.name, 'layout': layout, 'source_shape': source_shape, 'perm': list(perm)}

    def _preferred_layout(self, tensor, rank, depth=0):
        """The layout the first consumer of a tensor needs, looking through element-wise layers."""
        consumers = [n for n in self.model.graph.values() if tensor in n.inputs]
        if not consumers or depth > 32:
            return _identity(rank)
        consumer = consumers[0]
        kind = _layout_kind(consumer)
        if kind == 'channels_last':
            return _channels_last(rank) if rank <= 3 else _identity(rank)
        if kind in ('elementwise', 'concat', 'softmax'):
            out_rank = len(consumer.get_output_variable().shape)
            if out_rank == rank:
                return self._preferred_layout(consumer.outputs[0], rank, depth + 1)
        if kind == 'transpose':
            user_perm = consumer.get_attr('perm')
            needed = self._preferred_layout(consumer.outputs[0], rank, depth + 1)
            return tuple(user_perm[i] for i in needed)
        return _identity(rank)

    def _set_output_layout(self, index, out_name):
        perm = self.states.get(out_name)
        var = self.model.output_vars[out_name]
        rank = len(var.shape) if perm is None else len(perm)
        perm = perm if perm is not None else _identity(rank)
        layout = _contract(self.model, 'OutputLayout', out_name)
        source_shape = [var.shape[perm.index(i)] for i in range(rank)]
        if layout == 'source' and perm != _identity(rank):
            transpose = self._make_transpose(out_name, perm, _identity(rank), f'transpose_output_{out_name}', None)
            self.model.outputs[index] = transpose.outputs[0]
            if var.type.name == 'result_t':
                # The former output keeps its own type; result_t moves to the new output
                producer = next(n for n in self.model.graph.values() if out_name in n.outputs)
                var.type.name = f'layer{producer.index}_t'
            self.model.register_output_variable(transpose.outputs[0], transpose.get_output_variable())
            perm = _identity(rank)
        return {'name': out_name, 'layout': layout, 'source_shape': source_shape, 'perm': list(perm)}

    def _process(self, node):
        in_states = [self.states.get(i) for i in node.inputs]
        if any(s is None for s in in_states):
            raise RuntimeError(f'Layout inference reached {node.name} before all of its inputs.')
        out_shapes = [list(node.get_output_variable(o).shape) for o in node.outputs]
        out_rank = len(out_shapes[0])

        kind = _layout_kind(node)
        if kind is None:
            if node.class_name not in self.warned_classes:
                warnings.warn(
                    f'Layout inference has no rule for {node.class_name}; it is treated as needing the source layout.',
                    stacklevel=2,
                )
                self.warned_classes.add(node.class_name)
            kind = 'source'

        if kind == 'channels_last':
            rank = len(in_states[0])
            if rank > 3:
                raise NotImplementedError(
                    f'Layer {node.name}: {node.class_name} on tensors with more than three non-batch dimensions '
                    'is not supported.'
                )
            for i in range(len(node.inputs)):
                self._require(node, i, _channels_last(rank))
            if 'data_format' in node.attributes:
                node.set_attr('data_format', 'channels_last')
            out_state = _channels_last(out_rank)
        elif kind == 'transpose':
            # The user's permutation becomes a relabelling of the stored data, so the layer itself does nothing
            user_perm = node.get_attr('perm')
            out_state = tuple(user_perm.index(p) for p in in_states[0])
            self.nop_transposes.append(node)
        elif kind in ('elementwise', 'concat', 'softmax'):
            out_state = self._align_inputs(node, in_states, out_rank)
            if kind == 'concat':
                axis = node.get_attr('axis')
                axis = axis - 1 if axis > 0 else axis + out_rank
                node.set_attr('axis', out_state.index(axis) + 1)
            elif kind == 'softmax' and out_rank > 1:
                axis = node.get_attr('axis', -1)
                axis = axis - 1 if axis > 0 else axis + out_rank
                if out_state[-1] != axis:
                    target = tuple(a for a in out_state if a != axis) + (axis,)
                    self._require(node, 0, target)
                    out_state = target
                stored = [out_shapes[0][i] for i in out_state]
                node.set_attr('axis', -1)
                node.set_attr('n_outer', math.prod(stored[:-1]))
                node.set_attr('n_inner', 1)
            if out_state != _identity(out_rank):
                self._permute_parameters(node, out_shapes[0], out_state)
        else:
            for i, state in enumerate(in_states):
                self._require(node, i, _identity(len(state)))
            out_state = _identity(out_rank)

        for o, shape in zip(node.outputs, out_shapes):
            state = out_state if len(shape) == len(out_state) else _identity(len(shape))
            node.get_output_variable(o).shape = [shape[i] for i in state]
            self.states[o] = state

    def _align_inputs(self, node, in_states, out_rank):
        """Bring all full-rank inputs of an element-wise layer to the layout of the first one."""
        full_rank = [i for i, s in enumerate(in_states) if len(s) == out_rank and self._input_size(node, i) > 1]
        if not full_rank:
            return _identity(out_rank)
        target = in_states[full_rank[0]]
        for i in full_rank[1:]:
            self._require(node, i, target)
        return target

    def _input_size(self, node, idx):
        return math.prod(self.model.output_vars[node.inputs[idx]].shape)

    def _permute_parameters(self, node, source_shape, state):
        """Reorder the per-element parameters of an element-wise layer to match the stored order of its input."""
        if isinstance(node, FixedPointQuantizer):
            node.mask_kbi = tuple(self._permute_full_shape(t, state) for t in node.mask_kbi)
        elif isinstance(node, ApplyAlpha):
            for name in ('scale', 'bias'):
                var = node.get_weights(name)
                if var.data_length == math.prod(source_shape):
                    var.data = np.ascontiguousarray(np.reshape(var.data, source_shape).transpose(state))
                    var.shape = list(var.data.shape)

    @staticmethod
    def _permute_full_shape(t, state):
        t = np.asarray(t)
        rank = len(state)
        if t.ndim > rank + 1:
            return t
        t = t.reshape((1,) * (rank + 1 - t.ndim) + t.shape)
        return np.transpose(t, (0,) + tuple(i + 1 for i in state))

    def _require(self, node, idx, target):
        tensor = node.inputs[idx]
        state = self.states[tensor]
        if state == target:
            return
        key = (tensor, target)
        if key not in self.transposes:
            transpose = self._make_transpose(tensor, state, target, f'transpose_for_{node.name}', node)
            self.transposes[key] = transpose.outputs[0]
        node.inputs[idx] = self.transposes[key]

    def _make_transpose(self, tensor, state, target, name, before):
        model = self.model
        while name in model.graph:
            name += '_'
        perm = transpose_perm(state, target)
        transpose = model.make_node(Transpose, name, {'perm': perm}, [tensor])

        new_graph = {}
        for k, v in model.graph.items():
            if v is before:
                new_graph[name] = transpose
            new_graph[k] = v
        if before is None:
            new_graph[name] = transpose
        model.graph.clear()
        model.graph.update(new_graph)

        self.states[transpose.outputs[0]] = target
        if model.config.get_config_value('IOType') == 'io_stream':
            size = math.prod(transpose.get_output_variable().shape)
            warnings.warn(
                f'Layout inference inserted {name}, which buffers a whole tensor of {size} elements in io_stream.',
                stacklevel=2,
            )
        return transpose
