import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Concatenate, Layer, Merge, register_layer
from hls4ml.model.optimizer import OptimizerPass


class Broadcast(Layer):
    '''Inserted between layers for broadcasting.'''

    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = [f'N_SIZE_{i}_{self.index}' for i in range(1, len(shape) + 1)]
        self.add_output_variable(shape, dims)


broadcast_function_template = 'nnet::broadcast_stream<{input_t}, {output_t}, {config}>({input}, {output});'
broadcast_config_template = """struct config{index} : nnet::broadcast_config {{
    static const unsigned in_width = {in_width};
    static const unsigned in_height = {in_height};
    static const unsigned in_chan = {in_chan};
    static const unsigned out_width = {out_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_chan = {out_chan};
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
        params['in_chan'] = node.get_input_variable().shape[2]
        params['out_height'] = node.get_output_variable().shape[0]
        params['out_width'] = node.get_output_variable().shape[1]
        params['out_chan'] = node.get_output_variable().shape[2]

        return self.template.format(**params)


class BroadcastFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Broadcast, include_header=broadcast_include_list)
        self.template = broadcast_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def register_broadcast_stream(backend):
    # Register the layer types to the layer map
    register_layer('Broadcast', Broadcast)

    # Register the optimization passes
    backend.register_pass('broadcast_stream', BroadcastStream)

    # Register template passes
    backend.register_template(BroadcastConfigTemplate)
    backend.register_template(BroadcastFunctionTemplate)


class BroadcastStream(OptimizerPass):
    def match(self, node):
        if isinstance(node, Merge) and not isinstance(node, Concatenate):
            inp1 = node.get_input_variable(node.inputs[0])
            inp2 = node.get_input_variable(node.inputs[1])
            return inp1.shape != inp2.shape
        else:
            return False

    def transform(self, model, node):
        if model.config.backend.name not in ['Vivado'] or model.config.get_config_value('IOType') != 'io_stream':
            return False

        inp = [node.get_input_variable(inp_name) for inp_name in node.inputs]

        if np.prod(inp[0].shape) > np.prod(inp[1].shape):
            idx = 1
            attrs = {'target_shape': inp[0].shape}
        else:
            idx = 0
            attrs = {'target_shape': inp[1].shape}

        def supported_broadcast(inp_shape, target_shape):
            # Must be (H, W, C)
            if not len(inp_shape) == 3:
                return False
            # Supported: (1, 1, C) -> (H, W, C)
            if inp_shape[0] == inp_shape[1] == 1 and inp_shape[2] == target_shape[2]:
                return True
            # Supported: (H, W, 1) -> (H, W, C)
            if inp_shape[2] == 1 and inp_shape[0] == target_shape[0] and inp_shape[1] == target_shape[1]:
                return True
            return False

        brdcst_inp = node.inputs[idx]
        inp_shape = node.get_input_variable(brdcst_inp).shape
        target_shape = attrs['target_shape']
        if not supported_broadcast(inp_shape, target_shape):
            raise RuntimeError(
                f'Unsupported broadcast type for stream: {inp_shape} -> {target_shape};'
                + 'Only (1, 1, C) -> (H, W, C) and (H, W, 1) -> (H, W, C) currently supported'
            )
        brdcst_out = 'broadcast_' + brdcst_inp
        brdcst_layer = model.make_node('Broadcast', brdcst_out, attrs, [brdcst_inp].copy())
        model.insert_node(brdcst_layer, before=node, input_idx=idx)
        node.inputs[idx] = brdcst_out

        return True
