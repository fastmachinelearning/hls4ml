from ..optimizer import OptimizerPass

import hls4ml.model.hls_model as hls_model

class FuseBiasAdd(OptimizerPass):
    ''' Fuses BiasAdd into Dense layer (common in TF models). '''
    #TODO extend for Conv layers
    def match(self, node):
        is_match = node.__class__.__name__ == 'BiasAdd' and \
            node.get_input_node().__class__.__name__ == 'Dense'
        return is_match

    def transform(self, model, node):
        # Fuse BiasAdd into Dense layer
        dense_layer = node.get_input_node()
        dense_layer.get_weights('bias').data = node.get_weights('bias').data

        model.remove_node(node, rewire=True)

        return True