import numpy as np

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Layer, Merge, Reshape, register_layer
from hls4ml.backends import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate

class Repack(Layer):
    ''' Inserted between layers with different packing factors.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

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

class Broadcast(Layer):
    ''' Inserted between layers for broadcasting.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        self.add_output_variable(shape, dims)

broadcast_function_template = 'nnet::broadcast_stream<{input_t}, {output_t}, {config}>({input}, {output});'
broadcast_config_template = """struct config{index} : nnet::broadcast_config {{
    static const unsigned in_width = {in_width};
    static const unsigned in_height = {in_height};
    static const unsigned n_chan = {n_chan};
    static const unsigned n_dupl = {n_dupl};
}};\n"""
broadcast_include_list = ['nnet_utils/nnet_stream.h']

class BroadcastConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Broadcast)
        self.template = broadcast_config_template
    
    def format(self, node):
        params = self._default_config_params(node)
        params['in_height'] = node.get_input_variable().shape[0]
        params['in_width'] = node.get_input_variable().shape[1]
        params['n_chan'] = node.get_input_variable().shape[2]
        params['n_dupl'] = int(np.prod(node.get_output_variable().shape) / np.prod(node.get_input_variable().shape))

        return self.template.format(**params)

class BroadcastFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Broadcast, include_header=broadcast_include_list)
        self.template = broadcast_function_template
    
    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)

def register_repack_stream(backend):
    # Register the layer types to the layer map
    register_layer('Repack', Repack)
    register_layer('Broadcast', Broadcast)
    
    # Register the optimization passes
    backend.register_pass('remove_final_reshape', RemoveFinalReshape)
    backend.register_pass('reshape_stream', ReshapeStream)
    backend.register_pass('broadcast_stream', BroadcastStream)
    
    # Register template passes
    backend.register_template(RepackFunctionTemplate)
    backend.register_template(BroadcastConfigTemplate)
    backend.register_template(BroadcastFunctionTemplate)

class ReshapeStream(OptimizerPass):
    ''' Repacks stream for Reshape layer '''
    def match(self, node):
        # do not run optimizer pass for a flatten layer (1 output dimension)
        return isinstance(node, Reshape) and len(node.get_output_variable().shape) > 1

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False

        attrs = {
            'target_shape': node.get_attr('target_shape')
        }

        # Insert new Repack node instead of Reshape
        repack_layer = model.make_node(Repack, 'repack_' + node.name, attrs, node.inputs.copy())
        model.replace_node(node, repack_layer)

        return True

class BroadcastStream(OptimizerPass):
    def match(self, node):
        if isinstance(node, Merge):
            inp1 = node.get_input_variable(node.inputs[0])
            inp2 = node.get_input_variable(node.inputs[1])
            return inp1.shape != inp2.shape
        else:
            return False
        
    def transform(self, model, node):
        if model.config.backend.name not in ['Vivado'] or \
            model.config.get_config_value('IOType') != 'io_stream':
            return False
            
        inp1 = node.get_input_variable(node.inputs[0])
        inp2 = node.get_input_variable(node.inputs[1])
        if np.prod(inp1.shape) > np.prod(inp2.shape):
            idx = 1
            attrs = {
                'target_shape': inp1.shape
            }
        else:
            idx = 0
            attrs = {
                'target_shape': inp2.shape
            }
        brdcst_inp = node.inputs[idx]
        brdcst_out = 'broadcast_' + brdcst_inp
        brdcst_layer = model.make_node('Broadcast', brdcst_out, attrs, [brdcst_inp].copy())
        model.insert_node(brdcst_layer)
        node.inputs[idx] = brdcst_out

        return True

class RemoveFinalReshape(OptimizerPass):
    ''' Remove reshape if final layer '''
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
            print('WARNING: Final layer is a Reshape, which may incur a large resource cost for io_stream; consider removing it')
        return False
