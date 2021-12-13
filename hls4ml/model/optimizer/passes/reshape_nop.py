"""
This file is for the cases when a Reshape node can become effectively a NOP
"""
from copy import deepcopy
from hls4ml.model.hls_layers import Variable, Layer, register_layer
from hls4ml.model.optimizer import OptimizerPass

class InplaceVariable(Variable):
    """A reference variable. It does not output a definition_cpp--the layer using it outputs what is needed"""
    def __init__(self, shape, dim_names, var_name, atype, **kwargs):
        super().__init__(var_name, atype, **kwargs)
        self.shape = shape
        self.dim_names = dim_names

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        return None

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class ReshapeNop(Layer):
    """Effectively turns a reshape to a NOP when nothing needs to be done in the HLS code"""
    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        out_name = self.outputs[0]
        outtype = self.attributes['out_type']
        outtype.name = f'layer{self.index}_t'
        out = InplaceVariable(shape, dims, f'layer{self.index}_out', outtype)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

    def function_cpp(self):
        # for now don't register an actual function template but explicitly create it here
        invar = self.get_input_variable()
        outvar = self.get_output_variable()
        return [f'using {outvar.type.name} = {invar.type.name};',
                f'auto& {outvar.cppname} = {invar.cppname};']

    def config_cpp(self):
        return None

# Register the layer types to the layer map
register_layer('ReshapeNop', ReshapeNop)

class ReshapeToNop(OptimizerPass):
    ''' Changes Reshape to a ReshapeNop in certain cases '''
    def match(self, node):
        # do more filtering in the transfrom
        return node.__class__.__name__ == 'Reshape'

    def transform(self, model, node):
        # don't perform transform if io_stream and not flatten. (Another optimizer handles this case)
        if (len(node.get_output_variable().shape) > 1
            and model.config.get_config_value('IOType') == 'io_stream'):
            return False

        attrs = {
            'target_shape': node.get_attr('target_shape'),
            'out_type': deepcopy(node.get_input_variable().type)
        }

        # Insert new ReshapeNop node instead of Reshape
        newnode = model.make_node('ReshapeNop', 'Nop_' + node.name, attrs, node.inputs.copy())
        model.replace_node(node, newnode)

        return True
