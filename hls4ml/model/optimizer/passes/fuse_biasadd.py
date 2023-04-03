from hls4ml.model.layers import BiasAdd, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass


class FuseBiasAdd(OptimizerPass):
    '''Fuses BiasAdd into Dense/Conv2D layer (common in TF models).'''

    def match(self, node):
        return isinstance(node, BiasAdd) and isinstance(node.get_input_node(), (Dense, Conv1D, Conv2D))

    def transform(self, model, node):
        # Fuse BiasAdd into Dense layer
        dense_layer = node.get_input_node()
        dense_layer.get_weights('bias').data = node.get_weights('bias').data

        model.remove_node(node, rewire=True)

        return True
