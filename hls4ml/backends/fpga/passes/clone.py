import numpy as np

from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model.layers import Layer, register_layer, Reshape
from hls4ml.model.types import InplaceVariable
from hls4ml.backends.template import FunctionCallTemplate

class Clone(Layer):
    ''' Inserted after the layer whose output is used more than once.'''

    def initialize(self):
        inp = self.get_input_variable()
        for i, out_name in enumerate(self.outputs):
            self.add_output_variable(inp.shape, inp.dim_names, out_name=out_name, var_name='layer{index}_cpy' + str(i + 1))

clone_include_list = ['nnet_utils/nnet_stream.h']

class CloneFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Clone, include_header=clone_include_list)
        self.template = None # to be filled once number of clones known
    
    def format(self, node):
        params = self._default_function_params(node)
        for i, output in enumerate(node.outputs):
            params['output' + str(i + 1)] = node.variables[output].name
        
        if self.template is None:
            self.template = 'nnet::clone_stream<{input_t}, {output_t}, {size}>({input}, ' + \
                            ', '.join(['{output' + str(i + 1) + '}' for i in range(len(node.outputs))]) + \
                            ');'

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

        output_map = node.get_output_use_map()

        transformed = False
        for output in node.outputs:
            if len(output_map[output]) > 1:
                if len(output_map[output]) > 3:
                    print('WARNING: Cloning output {} of {} ({}) more than 3 times not currently supported'.format(output, node.__class__.__name__, node.name))
                    return False
                out_var = node.get_output_variable(output)
                attrs = {
                    'size' : np.prod(out_var.shape)
                }
                clone_layer = model.make_node(Clone, 'clone_' + node.name, attrs, [output], [output + '_cpy' + str(i + 1) for i in range(len(output_map[output]))])
                for i, layer in enumerate(output_map[output], 1):
                    idx = layer.inputs.index(output)
                    layer.inputs[idx] = output + '_cpy' + str(i)
                    if isinstance(layer, Reshape):
                        proxy = clone_layer.get_output_variable(output + '_cpy' + str(i))
                        current_out = layer.get_output_variable()
                        shape = current_out.shape
                        dims = [f'N_SIZE_{j}_{layer.index}' for j in range(1, len(shape) + 1)]
                        new_out = InplaceVariable(shape, dims, proxy)
                        layer.set_attr(layer.outputs[0], new_out)
                model.insert_node(clone_layer)
                transformed = True
        
        return transformed
