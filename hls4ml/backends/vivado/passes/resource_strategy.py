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
)
from hls4ml.model.optimizer import OptimizerPass


class ApplyResourceStrategy(OptimizerPass):
    '''Transposes the weights to use the dense_resource matrix multiply routine'''

    def match(self, node):
        node_matches = isinstance(node, (Dense, Conv1D, SeparableConv1D, Conv2D, SeparableConv2D, LSTM, GRU, Bidirectional))
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
                node.weights[f'{d}_weight'].data = np.transpose(node.weights[f'{d}_weight'].data)
                node.weights[f'{d}_recurrent_weight'].data = np.transpose(node.weights[f'{d}_recurrent_weight'].data)
        elif isinstance(node, (LSTM, GRU)):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)
        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')

        node.set_attr('_weights_transposed', True)

        return False
