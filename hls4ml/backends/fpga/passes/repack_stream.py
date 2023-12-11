import numpy as np

from hls4ml.backends.template import FunctionCallTemplate
from hls4ml.model.layers import Layer, Reshape, register_layer
from hls4ml.model.optimizer import OptimizerPass


class Repack(Layer):
    '''Inserted between layers with different packing factors.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = [f'N_SIZE_{i}_{self.index}' for i in range(1, len(shape) + 1)]

        self.add_output_variable(shape, dims)


repack_function_template = 'nnet::repack_stream<{input_t}, {output_t}, {size}>({input}, {output});'
repack_include_list = ['nnet_utils/nnet_stream.h']


class RepackFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Repack, include_header=repack_include_list)
        self.template = repack_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['size'] = np.prod(node.get_output_variable().shape)

        return self.template.format(**params)


def register_repack_stream(backend):
    # Register the layer types to the layer map
    register_layer('Repack', Repack)

    # Register the optimization passes
    backend.register_pass('reshape_stream', ReshapeStream)

    # Register template passes
    backend.register_template(RepackFunctionTemplate)


class ReshapeStream(OptimizerPass):
    '''Repacks stream for Reshape layer'''

    def match(self, node):
        # do not run optimizer pass for a flatten layer (1 output dimension)
        return isinstance(node, Reshape) and len(node.get_output_variable().shape) > 1

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False

        attrs = {'target_shape': node.get_attr('target_shape')}

        # Insert new Repack node instead of Reshape
        repack_layer = model.make_node(Repack, 'repack_' + node.name, attrs, node.inputs.copy())
        # As result_t attribute is not honored by type conversion, set it manually here
        repack_layer.attributes[repack_layer.name].type = node.attributes[node.name].type
        model.replace_node(node, repack_layer)

        return True
