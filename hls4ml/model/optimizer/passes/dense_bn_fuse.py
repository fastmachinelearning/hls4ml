from hls4ml.model.optimizer import OptimizerPass

class FuseDenseAndBatchNormalization(OptimizerPass):
    def match(self, node):
        is_match = node.__class__.__name__ == 'BatchNormalization' and \
            node.get_input_node().__class__.__name__ == 'Dense' and \
            node.get_input_node().get_attr('weight_quantizer') is None and \
            node.get_input_node().get_attr('bias_quantizer') is None
        return is_match

    def transform(self, model, node):
        # Fuse weight and bias of Dense layer with BN values
        dense_node = node.get_input_node()

        dense_weight = dense_node.weights['weight']
        dense_bias = dense_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        if dense_node.get_attr('strategy') != 'large':
            fused_weight = bn_scale.data * dense_weight.data
        else:
            fused_weight = (bn_scale.data * dense_weight.data.T).T
        fused_bias = bn_scale.data * dense_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        dense_weight.data = fused_weight
        dense_bias.data = fused_bias
        if not dense_node.get_attr('use_bias', True):
            dense_bias.update_precision(bn_bias.type.precision)

        return True
