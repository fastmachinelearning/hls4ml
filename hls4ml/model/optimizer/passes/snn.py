from hls4ml.model.layers import IFNeuron, LIFNeuron, SNNReadout
from hls4ml.model.optimizer import ModelOptimizerPass


class PropagateSNNReadoutWindowSize(ModelOptimizerPass):
    """
    Propagate fixed-window SNN readout length to upstream neuron layers.
    """

    name = 'propagate_snn_readout_window_size'

    def __init__(self):
        pass

    def transform(self, model):
        readouts = [
            node
            for node in model.graph.values()
            if isinstance(node, SNNReadout) and node.get_attr('state_reset_policy', 'fixed_window') == 'fixed_window'
        ]
        if len(readouts) == 0:
            return False

        window_size = readouts[0].get_attr('window_size', 0)
        changed = False
        for node in model.graph.values():
            if isinstance(node, (IFNeuron, LIFNeuron)) and node.get_attr('window_size') != window_size:
                node.set_attr('window_size', window_size)
                changed = True

        return changed
