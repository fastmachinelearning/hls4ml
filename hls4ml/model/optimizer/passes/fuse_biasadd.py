from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model.hls_layers import BiasAdd, Dense, Conv2D

class FuseBiasAdd(OptimizerPass):
    ''' Fuses BiasAdd into Dense/Conv2D layer (common in TF models). '''
    def match(self, node):
        is_match = isinstance(node, BiasAdd) and \
            (isinstance(node.get_input_node(), Dense) or
            isinstance(node.get_input_node(), Conv2D))
        return is_match

    def transform(self, model, node):
        # Fuse BiasAdd into Dense layer
        dense_layer = node.get_input_node()
        dense_layer.get_weights('bias').data = node.get_weights('bias').data

        model.remove_node(node, rewire=True)

        return True
