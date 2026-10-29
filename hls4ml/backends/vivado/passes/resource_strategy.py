from warnings import warn

import numpy as np

from hls4ml.model.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
)
from hls4ml.model.optimizer import OptimizerPass


def _replace_data(var, data):
    var.data = data
    var.shape = list(data.shape)
    var.data_length = int(np.prod(data.shape))


def _lay_out_recurrent_weights(node, prefix=''):
    """Lay out the weights of a recurrent layer in the order the cell of the Resource strategy reads them (see
    nnet_recurrent.h). A PyTorch GRU has its gates in the order r, z, n, which becomes the order of Keras, z, r, h."""

    n_state, n_columns = node.weights[prefix + 'recurrent_weight'].data.shape
    n_gates = n_columns // n_state
    order = [1, 0, 2] if n_gates == 3 and node.get_attr('pytorch', False) else list(range(n_gates))
    n_unit_par = node.get_attr(f'{prefix}n_unit_par')
    n_cycles = node.get_attr(f'{prefix}n_cycles')

    for name, lanes in (('weight', 'n_lanes_in'), ('recurrent_weight', 'n_lanes_state')):
        var = node.weights[prefix + name]
        n_rows = var.data.shape[0]
        n_lanes = node.get_attr(prefix + lanes)
        if getattr(var, 'storage', '').lower() == 'bram':
            warn(
                f'Layer "{node.name}": with BramFactor, the weights {var.name} become {n_unit_par * n_gates * n_lanes} '
                'separate memory ports, one for each value the Resource strategy reads in a cycle.'
            )
        padded = np.zeros((n_cycles * n_lanes, n_gates, n_state), dtype=var.data.dtype)
        padded[:n_rows] = var.data.reshape(n_rows, n_gates, n_state)[:, order, :]
        # (cycle, lane, gate, group, unit) -> (group, cycle, unit, gate, lane)
        padded = padded.reshape(n_cycles, n_lanes, n_gates, n_state // n_unit_par, n_unit_par)
        _replace_data(var, np.ascontiguousarray(padded.transpose(3, 0, 4, 2, 1)).reshape(-1))

    for name in ('bias', 'recurrent_bias'):
        var = node.weights[prefix + name]
        _replace_data(var, np.ascontiguousarray(var.data.reshape(n_gates, -1)[order].T).reshape(-1))


class ApplyResourceStrategy(OptimizerPass):
    """Transposes the weights to use the dense_resource matrix multiply routine, or lays out those of a recurrent layer
    for the cell of the Resource strategy."""

    def match(self, node):
        node_matches = isinstance(
            node, (Dense, Conv1D, SeparableConv1D, Conv2D, SeparableConv2D, SimpleRNN, LSTM, GRU, Bidirectional)
        )
        is_resource_strategy = node.get_attr('strategy', '').lower() in ['resource', 'resource_unrolled']
        already_transformed = node.get_attr('_weights_transposed', False) is True
        return node_matches and is_resource_strategy and not already_transformed

    def transform(self, model, node):
        if isinstance(node, Dense):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
        elif isinstance(node, Conv1D):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[2, 0, 1])  # (W,C,F) => (F,W,C)
        elif isinstance(node, SeparableConv1D):
            node.weights['depthwise'].data = np.transpose(
                node.weights['depthwise'].data, axes=[2, 0, 1]
            )  # (W,C,F) => (F,W,C)
            node.weights['pointwise'].data = np.transpose(
                node.weights['pointwise'].data, axes=[2, 0, 1]
            )  # (W,C,F) => (F,W,C)
        elif isinstance(node, Conv2D):
            node.weights['weight'].data = np.transpose(
                node.weights['weight'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
        elif isinstance(node, SeparableConv2D):
            node.weights['depthwise'].data = np.transpose(
                node.weights['depthwise'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
            node.weights['pointwise'].data = np.transpose(
                node.weights['pointwise'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
        elif isinstance(node, (Bidirectional)):
            for d in ['forward', 'backward']:
                _lay_out_recurrent_weights(node, prefix=f'{d}_')
        elif isinstance(node, (SimpleRNN, LSTM, GRU)):
            if node.get_attr('strategy').lower() == 'resource':
                _lay_out_recurrent_weights(node)
            else:
                node.weights['weight'].data = np.transpose(node.weights['weight'].data)
                node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)
        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')

        node.set_attr('_weights_transposed', True)

        return False
