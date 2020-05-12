from ..optimizer import OptimizerPass
from ....model.hls_model import IntegerPrecisionType, FixedPrecisionType

class OutputRoundingSaturationMode(OptimizerPass):
    '''
    Set the Rounding and Saturation mode of the output (and accumulator, if applicable)
    of the layers specific in layer list.
    The layer list is empty by default.
    To specify which layer to apply this pass to, perform e.g.:
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Dense', 'Activation', 'BatchNormalization']
    The Rounding and Saturation modes are 'None' by default (so use the compiler defaults)
    To set which mode to use:
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND_CONV'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    '''

    layers = [] 
    rounding_mode = None 
    saturation_mode = None 

    def match(self, node):
        return node.__class__.__name__ in self.layers

    def transform(self, model, node):
        oldtype = node.get_output_variable().type.precision
        print(type(oldtype))
        if isinstance(oldtype, IntegerPrecisionType):
            newprecision = IntegerPrecisionType(oldtype.width, oldtype.signed, rounding_mode, saturation_mode)
        elif isinstance(oldtype, FixedPrecisionType):
            newtype = FixedPrecisionType(oldtype.width, oldtype.integer, oldtype.signed, rounding_mode, saturation_mode)
        else: # in case the precision is a string
            newtype = self.precision_string_modify(oldtype)
        print(type(newtype))
        node.get_output_variable().type.precision = newtype
        if node.get_attr('accum_t') is not None:
            node.set_attr('accum_t', newtype)

    def precision_string_modify(self, pstr):
        # For when the type is a string not an Type
        mode = ''
        if self.rounding_mode is not None:
            mode += ',' + self.rounding_mode
        if self.saturation_mode is not None:
            mode += ',' + self.saturation_mode
        mode += '>'
        pstr = pstr.replace('>', mode)
        return pstr


