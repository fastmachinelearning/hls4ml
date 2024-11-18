from math import prod

from hls4ml.backends.template import FunctionCallTemplate
from hls4ml.model.layers import Layer, register_layer
from hls4ml.model.optimizer import OptimizerPass


class Clone(Layer):
    '''Inserted after the layer whose output is used more than once.'''

    def initialize(self):
        inp = self.get_input_variable()
        for i, out_name in enumerate(self.outputs):
            self.add_output_variable(inp.shape, inp.dim_names, out_name=out_name, var_name='layer{index}_cpy' + str(i + 1))


clone_include_list = ['nnet_utils/nnet_stream.h']


class CloneFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Clone, include_header=clone_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        for i, _output in enumerate(node.outputs):
            params['output' + str(i + 1)] = node.variables[node.outputs[i]].name

        template = (
            'nnet::clone_stream<{input_t}, {output_t}, {size}>({input}, '
            + ', '.join(['{output' + str(i + 1) + '}' for i in range(len(node.outputs))])
            + ');'
        )

        return template.format(**params)


def register_clone(backend):
    # Register the layer types to the layer map
    register_layer('Clone', Clone)

    # Register the optimization passes
    backend.register_pass('clone_output', CloneOutput)

    # Register template passes
    backend.register_template(CloneFunctionTemplate)


class CloneOutput(OptimizerPass):
    '''Clones streams that are used multiple times'''

    def match(self, node):
        # We may have already inserted the Clone layer
        if isinstance(node, Clone):
            return False

        # Not needed for io_parallel
        io_type = node.model.config.get_config_value('IOType')
        if io_type != 'io_stream':
            return False

        # Check if the output is used more than once
        output_map = node.get_output_use_map()
        in_output = node.name in node.model.outputs
        for output in node.outputs:
            if len(output_map[output]) + in_output > 1:
                # model output also need a stream
                return True

        return False

    def transform(self, model, node):

        output_map = node.get_output_use_map()
        in_output = node.name in node.model.outputs

        transformed = False
        for output in node.outputs:
            n_outputs = len(output_map[output]) + in_output
            if n_outputs == 1:
                continue
            if n_outputs > 3:
                msg = f'ERROR: Cloning output {output} of {node.class_name}\
                      ({node.name}) more than 3 times not currently supported'
                raise ValueError(msg)

            out_var = node.get_output_variable(output)
            attrs = {'size': prod(out_var.shape)}

            init_stream_idx = 1
            if in_output:
                # If the value is used as output, add one extra stream
                idx = node.model.outputs.index(node.name)
                node.model.outputs[idx] = node.name + '_cpy1'
                init_stream_idx = 2
            for i, layer in enumerate(output_map[output], init_stream_idx):
                idx = layer.inputs.index(output)
                layer.inputs[idx] = output + f'_cpy{i}'

            clone_layer: Clone = model.make_node(
                Clone,
                'clone_' + node.name,
                attrs,
                [output],
                [output + '_cpy' + str(i + 1) for i in range(n_outputs)],
            )
            for i in range(n_outputs):
                key = output + '_cpy' + str(i + 1)
                clone_layer.attributes[key].type = node.attributes['result_t']
            model.insert_node(clone_layer)
            transformed = True

        return transformed
