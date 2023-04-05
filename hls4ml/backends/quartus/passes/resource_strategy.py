import numpy as np

from hls4ml.model.layers import GRU, LSTM, Conv1D, Conv2D, Dense, SimpleRNN
from hls4ml.model.optimizer import OptimizerPass


class ApplyResourceStrategy(OptimizerPass):
    '''Transposes the weights to use the dense_resource matrix multiply routine'''

    def match(self, node):
        node_matches = isinstance(node, (Dense, Conv1D, Conv2D, GRU, LSTM, SimpleRNN))
        is_resource_strategy = (
            True  # node.get_attr('strategy', '').lower() == 'resource' -> Quartus only supportr Resource strategy
        )
        already_transformed = node.get_attr('_weights_transposed', False) is True
        return node_matches and is_resource_strategy and not already_transformed

    def transform(self, model, node):
        if isinstance(node, Dense) and not node.model.config.get_compression(node):
            rf = node.get_attr('reuse_factor')
            bf = int((node.attributes['n_in'] * node.attributes['n_out']) / rf)
            bf_rounded = int(pow(2, np.ceil(np.log2(bf))))
            rf_rounded = int(pow(2, np.ceil(np.log2(rf))))

            node.weights['weight'].data = np.transpose(node.weights['weight'].data).flatten()

            if node.attributes['n_in'] * node.attributes['n_out'] > 2048 and rf_rounded != rf:
                node.set_attr('rfpad', rf_rounded - rf)
                node.set_attr('bfpad', bf_rounded - bf)

                temp = np.empty([bf_rounded, rf_rounded])
                for i in range(rf_rounded):
                    for j in range(bf_rounded):
                        if i < rf and j < bf:
                            w_index = i + rf * j
                            temp[j][i] = node.weights['weight'].data[w_index]
                        else:
                            temp[j][i] = 0
                node.weights['weight'].data = temp.flatten()
                node.weights['weight'].data_length = node.weights['weight'].data.size

        elif isinstance(node, Conv1D):
            # (W,C,F) => (F,W,C)
            # IMPORTANT - This format only works with im2col convolution
            #           - Future commits add new optimizers that further transpose THIS format to a format
            #                 useful for Winograd's minimal filtering algorithm
            node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[2, 0, 1])

        elif isinstance(node, Conv2D):
            # (H,W,C,F) => (F,H,W,C)
            # IMPORTANT - This format only works with im2col convolution
            #           - Future commits add new optimizers that further transpose THIS format to a format
            #                 useful for Winograd's minimal filtering algorithm
            node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[3, 0, 1, 2])

        elif isinstance(node, GRU):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)

        elif isinstance(node, SimpleRNN):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)

        elif isinstance(node, LSTM):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)

            for weight_type in ['i', 'f', 'c', 'o']:
                node.weights[f'weight_{weight_type}'].data = np.transpose(node.weights[f'weight_{weight_type}'].data)
                node.weights[f'recurrent_weight_{weight_type}'].data = np.transpose(
                    node.weights[f'recurrent_weight_{weight_type}'].data
                )

        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')
        node.set_attr('_weights_transposed', True)
        return False
