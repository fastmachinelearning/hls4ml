from hls4ml.model.layers import Reshape
from hls4ml.model.optimizer import OptimizerPass


class RemoveFinalReshape(OptimizerPass):
    '''Remove reshape if final layer'''

    def match(self, node):
        # match if reshape is final node
        return isinstance(node, Reshape) and not node.get_output_nodes()

    def transform(self, model, node):
        if model.config.get_config_value('IOType') == 'io_parallel':
            print('WARNING: Final layer is a Reshape, which does not affect the output for io_parallel; removing it')
            # remove, but don't rewire because it's the output layer
            model.remove_node(node, rewire=False)
            return True
        elif model.config.get_config_value('IOType') == 'io_stream':
            print(
                'WARNING: Final layer is a Reshape, which may incur a large resource cost for io_stream; '
                'consider removing it'
            )
        return False
