import numpy as np

from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model.layers import Layer, register_layer
from hls4ml.backends import get_backend
from hls4ml.backends.template import FunctionCallTemplate

class Clone(Layer):
    ''' Inserted after the layer whose output is used more than once.'''

    def initialize(self):
        inp = self.get_input_variable()
        self.add_output_variable(inp.shape, inp.dim_names, out_name=self.outputs[0], var_name='layer{index}_cpy1')
        self.add_output_variable(inp.shape, inp.dim_names, out_name=self.outputs[1], var_name='layer{index}_cpy2')

clone_function_template = 'nnet::clone_stream<{input_t}, {output_t}, {size}>({input}, {output1}, {output2});'
clone_include_list = ['nnet_utils/nnet_stream.h']

class CloneFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Clone, include_header=clone_include_list)
        self.template = clone_function_template
    
    def format(self, node):
        params = self._default_function_params(node)
        params['output1'] = node.variables[node.outputs[0]].name
        params['output2'] = node.variables[node.outputs[1]].name

        return self.template.format(**params)

def register_clone(backend):
    # Register the layer types to the layer map
    register_layer('Clone', Clone)

    # Register the optimization passes
    backend.register_pass('clone_output', CloneOutput)

    # Register template passes
    backend.register_template(CloneFunctionTemplate)

class CloneOutput(OptimizerPass):
    ''' Clones streams that are used multiple times '''
    def match(self, node):
        # We may have already inserted the Clone layer
        if isinstance(node, Clone):
            return False

        return True

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False

        output_map = {}
        for output in node.outputs:
            output_map[output] = []
            for layer in model.get_layers():
                for inp in layer.inputs:
                    if output == inp:
                        output_map[output].append(layer)

        transformed = False
        for output in node.outputs:
            if len(output_map[output]) > 1:
                if len(output_map[output]) > 2:
                    print('WARN: Cannot clone output {} of {} ({})'.format(output, node.class_name, node.name))
                    return False
                out_var = node.get_output_variable(output)
                for i, layer in enumerate(output_map[output], 1):
                    attrs = {
                        'size' : np.prod(out_var.shape)
                    }
                    idx = layer.inputs.index(output)
                    layer.inputs[idx] = output + '_cpy' + str(i)
                clone_layer = model.make_node(Clone, 'clone_' + node.name, attrs, [output], [output + '_cpy1', output + '_cpy2'])
                model.insert_node(clone_layer)
                transformed = True
        
        return transformed
