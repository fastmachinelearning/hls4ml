import numpy as np

from hls4ml.model.hls_types import Variable, TensorVariable, PackedType

class ArrayVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, pragma='partition', **kwargs):
        super(ArrayVariable, self).__init__(shape, dim_names, var_name, type_name, precision, **kwargs)
        self.pragma = pragma

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    @classmethod
    def from_variable(cls, tensor_var, pragma='partition', **kwargs):
        return cls(tensor_var.shape, tensor_var.dim_names, var_name=tensor_var.name, type_name=tensor_var.type.name, precision=tensor_var.type.precision, pragma=pragma)


class StreamVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, n_pack=1, depth=0, **kwargs):
        super(StreamVariable, self).__init__(shape, dim_names, var_name, type_name, precision, **kwargs)
        self.type = PackedType(type_name, precision, shape[-1], n_pack, **kwargs)
        if depth == 0:
            depth = np.prod(shape) // shape[-1]
        self.pragma = ('stream', depth)

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    @classmethod
    def from_variable(cls, tensor_var, n_pack=1, depth=0, **kwargs):
        return cls(tensor_var.shape, tensor_var.dim_names, var_name=tensor_var.name, type_name=tensor_var.type.name, precision=tensor_var.type.precision, n_pack=n_pack, depth=depth, **kwargs)
