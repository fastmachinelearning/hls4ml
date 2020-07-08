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
        model.backend.bn_weight_fuse(model, node)

        return True
