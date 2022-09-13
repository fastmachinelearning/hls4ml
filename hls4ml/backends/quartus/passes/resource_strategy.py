import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import LSTM, Dense, GRU, SimpleRNN

class ApplyResourceStrategy(OptimizerPass):
    ''' Transposes the weights to use the dense_resource matrix multiply routine '''
    def match(self, node):
        node_matches = isinstance(node, (Dense, SimpleRNN, LSTM, GRU))
        is_resource_strategy = True # node.get_attr('strategy', '').lower() == 'resource' ... Quartus only supports resource strategy
        already_transformed = node.get_attr('_weights_transposed', False) == True
        return node_matches and is_resource_strategy and not already_transformed

    def transform(self, model, node):
        if isinstance(node, Dense) and not node.model.config.get_compression(node):
            rf = node.get_attr('reuse_factor')
            bf = int((node.attributes['n_in']*node.attributes['n_out'])/rf)
            bf_rounded = int(pow(2, np.ceil(np.log2(bf))))
            rf_rounded = int(pow(2, np.ceil(np.log2(rf))))

            node.weights['weight'].data = np.transpose(node.weights['weight'].data).flatten()

            if(node.attributes['n_in']*node.attributes['n_out'] > 2048 and rf_rounded != rf):
                node.set_attr('rfpad', rf_rounded-rf)
                node.set_attr('bfpad', bf_rounded-bf)

                temp = np.empty([bf_rounded, rf_rounded])
                for i in range(rf_rounded):
                    for j in range (bf_rounded):
                        if (i < rf and j < bf):
                            w_index = i + rf * j
                            temp[j][i] = node.weights['weight'].data[w_index]
                        else:
                            temp[j][i] = 0
                node.weights['weight'].data = temp.flatten()
                node.weights['weight'].data_length = node.weights['weight'].data.size
        
        elif isinstance(node, SimpleRNN):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)

        elif isinstance(node, LSTM):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)

            for weight_type in ['i', 'f', 'c', 'o']:
                node.weights[f'weight_{weight_type}'].data = np.transpose(node.weights[f'weight_{weight_type}'].data)
                node.weights[f'recurrent_weight_{weight_type}'].data = np.transpose(node.weights[f'recurrent_weight_{weight_type}'].data)
        
        elif isinstance(node, GRU):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)        
        else:
            raise Exception('Unexpected layer {} with resource strategy'.format(node.class_name))

        node.set_attr('_weights_transposed', True)
        return False 

