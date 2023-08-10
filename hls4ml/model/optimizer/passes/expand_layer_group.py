from hls4ml.model.layers import Input, LayerGroup
from hls4ml.model.optimizer import OptimizerPass


class ExpandLayerGroup(OptimizerPass):
    '''Expands LayerGroup (a nested model) into the parent model.'''

    def match(self, node):
        return isinstance(node, LayerGroup)

    def transform(self, model, node):
        layer_list = node.get_attr('layer_list')

        # We'll keep track of inserted Input nodes to remove later
        inserted_input_nodes = []

        for i, layer in enumerate(layer_list):
            kind = layer['class_name']
            name = layer['name']
            inputs = layer.get('inputs', [])
            outputs = layer.get('outputs', [])

            if name in model.graph.keys():
                raise Exception(f'Layer names must be unique: "{name}" already found in the model graph.')

            if len(inputs) == 0:
                if kind in ['InputLayer', 'Input']:
                    inputs = node.inputs.copy()
                else:
                    inputs = model.graph[layer_list[i - 1]['name']].outputs.copy()
            if len(outputs) == 0:
                outputs = [name]

            new_node = model.make_node(kind, name, layer, inputs, outputs)
            model.insert_node(new_node)
            if isinstance(new_node, Input):
                inserted_input_nodes.append(new_node)

        rewire = not node.outputs[0] in model.outputs

        model.remove_node(node, rewire)

        for input_node in inserted_input_nodes:
            model.remove_node(input_node, rewire=True)

        return True
