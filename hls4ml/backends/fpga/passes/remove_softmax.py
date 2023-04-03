from hls4ml.model.layers import Softmax
from hls4ml.model.optimizer.optimizer import OptimizerPass


class SkipSoftmax(OptimizerPass):
    def match(self, node):
        is_softmax = isinstance(node, Softmax)
        remove_softmax = node.get_attr('skip', False)
        return is_softmax and remove_softmax

    def transform(self, model, node):
        model.remove_node(node, rewire=True)
        return True
