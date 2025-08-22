"""
This file contains optimizers to add layers to extract and merge the sidebands. This is useful
for the accelerator flow when using io_stream.

Warning:  current version only works for network with single inputs and outputs.

"""

import warnings
from collections import OrderedDict

from hls4ml.backends.oneapi_accelerator.oneapi_accelerator_layers import SidebandExtraction, SidebandMerging
from hls4ml.model.layers import Input
from hls4ml.model.optimizer import OptimizerPass


class ExtractSideband(OptimizerPass):
    """Add a layer after the input to extract the sideband signals."""

    def match(self, node):
        if not (isinstance(node, Input) and node.model.config.get_config_value('IOType') == 'io_stream'):
            return False
        # now check that not already converted
        output_nodes = node.get_output_nodes()
        if len(output_nodes) == 1 and isinstance(output_nodes[0], SidebandExtraction):
            # already transformed
            return False
        return True

    def transform(self, model, node):
        if len(model.inputs) > 1:
            warnings.warn('Current sideband extraction scheme only tested on models with one input', stacklevel=1)

        attributes = {'input_shape': node.get_attr('input_shape')}
        new_node = model.make_node(
            SidebandExtraction,
            f'{node.name}_extract_sb',
            attributes,
            inputs=[node.outputs[0]],
            outputs=[f'{node.name}_extract_sb', 'sideband'],
        )
        model.insert_node(new_node)
        return True


class MergeSideband(OptimizerPass):
    """Add a layer after the last layer to merge the sideband signals."""

    def match(self, node):
        for node_out in node.outputs:
            if node_out in node.model.outputs:  # if the node output is a model output
                return True
        return False

    def transform(self, model, node):
        if len(model.outputs) > 1:
            warnings.warn('Current sideband extraction scheme only tested on models with one output', stacklevel=1)

        attributes = {}

        inputs = [out for out in node.outputs if out in model.outputs]

        if len(inputs) != 1:
            raise RuntimeError('Unsupported number of outputs found')

        inputs.append('sideband')

        new_name = f'{node.name}_merge_sb'
        new_node = model.make_node(SidebandMerging, new_name, attributes, inputs=inputs)

        # note that model.insert_node fails here because of the two input nodes, so using a custom version below
        model.outputs[0] = new_name

        new_graph = OrderedDict()
        for k, v in model.graph.items():
            new_graph[k] = v
            if k == node.name:
                new_graph[new_node.name] = new_node

        model.graph = new_graph

        return True
