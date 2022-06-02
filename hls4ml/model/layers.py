import numpy as np
import six

from hls4ml.model.types import NamedType
from hls4ml.model.types import TensorVariable, WeightVariable, CompressedWeightVariable, ExponentWeightVariable, InplaceVariable
from hls4ml.model.types import IntegerPrecisionType, FixedPrecisionType, ExponentPrecisionType
from hls4ml.model.types import find_minimum_width

from hls4ml.model.attributes import Attribute, TypeMapping, VariableMapping, WeightAttribute, TypeAttribute, ChoiceAttribute, WeightMapping
from hls4ml.model.attributes import AttributeDict, AttributeMapping

# TODO move this to some utility module
class classproperty(object):
    def __init__(self, func):
        self.func = func
    
    def __get__(self, obj, owner):
        return self.func(owner)

class Layer(object):
    _expected_attributes = [
        Attribute('index'),

        TypeAttribute('accum'),
        TypeAttribute('result'),
    ]

    @classproperty
    def expected_attributes(cls):
        """ Returns the expected attributes of a class. """
        all_attributes = []
        for base_cls in reversed(cls.mro()): # Iterate over all base classes in the hierarchy
            if cls == base_cls: # Skip adding attributes from self
                continue
            if hasattr(base_cls, '_expected_attributes'): # Only consider classes with '_expected_attributes' defined
                all_attributes.extend(base_cls._expected_attributes)
        all_attributes.extend(cls._expected_attributes)
        return all_attributes

    def __init__(self, model, name, attributes, inputs, outputs=None):
        if name == 'input':
            raise RuntimeError("No model layer should be named 'input' because that is a reserved;" + \
                               "layer name in ModelGraph; Please rename the layer in your model")
        self.model = model
        self.name = name
        self.index = model.next_layer()
        self.inputs = inputs
        self.outputs = outputs
        if self.outputs is None:
            self.outputs = [self.name]

        self.attributes = AttributeDict(self)
        self.attributes.update(attributes)

        self.set_attr('index', self.index)

        self.weights = WeightMapping(self.attributes)
        self.variables = VariableMapping(self.attributes)
        self.types = TypeMapping(self.attributes)

        accum_t = NamedType(*reversed(self.model.config.get_precision(self, 'accum')))
        self.set_attr('accum_t', accum_t)

        layer_config = self.model.config.get_layer_config(self)
        for config_key, config_value in layer_config.items():
            if config_key in self.attributes:
                print('WARNING: Config parameter "{}" overwrites an existing attribute in layer "{}" ({})'.format(config_key, self.name, self.class_name))
            if config_key.endswith('_t') and isinstance(config_value, str): #TODO maybe move this to __setitem__ of AttributeDict?
                precision = self.model.config.backend.convert_precision_string(config_value)
                config_value = NamedType(self.name + config_key, precision)
            self.attributes[config_key] = config_value

        self.initialize()
        self._validate_attributes()

    @property
    def class_name(self, include_wrapped=False):
        if include_wrapped:
            return self.__class__.__name__
        else:
            if hasattr(self, '_wrapped'):
                return self.__class__.__bases__[0].__name__
            else:
                return self.__class__.__name__

    def initialize(self):
        raise NotImplementedError

    def set_attr(self, key, value):
        self.attributes[key] = value

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def _validate_attributes(self):
        all_attributes = {}
        for attr in self.expected_attributes:
            all_attributes[attr.name] = attr
        
        # Validate existing attributes
        for attr_name, attr_value in self.attributes.items():
            exp_attr = all_attributes.pop(attr_name, None)
            if exp_attr is not None:
                if not exp_attr.validate_value(attr_value):
                    raise Exception('Unexpected value of attribute "{}" of layer "{}" ({}). Expected {}, got {} ({})'
                        .format(attr_name, self.name, self.class_name, exp_attr.value_type, type(attr_value), attr_value))
            else:
                pass # TODO layer contains attribute that is not expected. we can log this for debugging
        
        # If any expected attributes remain, try adding their default values
        for attr_name, attr in all_attributes.items():
            if attr.default is not None:
                self.set_attr(attr_name, attr.default)
            else:
                raise Exception('Attribute "{}" of layer {} ({}) not set and no default value is specified.'.format(attr_name, self.name, self.class_name))

    def get_input_node(self, input_name=None):
        if input_name is not None:
            nodes = [node for node in self.model.graph.values() if input_name in node.outputs]
            if len(nodes) == 0:
                return None
            else:
                return nodes[0]
        else:
            return self.model.graph.get(self.inputs[0])

    def get_input_variable(self, input_name=None):
        if input_name is not None:
            return self.model.get_layer_output_variable(input_name)
        else:
            return self.model.get_layer_output_variable(self.inputs[0])

    def get_output_nodes(self, output_name=None):
        if output_name is None:
            output_name = self.outputs[0]
        return [node for node in self.model.graph.values() if node.inputs[0] == output_name]

    def get_output_variable(self, output_name=None):
        if output_name is not None:
            return self.variables[output_name]
        else:
            return next(iter(self.variables.values()))

    def get_weights(self, var_name=None):
        if var_name:
            return self.weights[var_name]

        return self.weights.values()

    def get_variables(self):
        return self.variables.values()

    def add_output_variable(self, shape, dim_names, out_name=None, var_name='layer{index}_out', type_name='layer{index}_t', precision=None):
        if out_name is None:
            out_name = self.outputs[0]

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var='result')

        out = TensorVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, index=self.index)

        self.set_attr(out_name, out)

    def add_weights(self, quantizer=None, compression=False):
        data = self.model.get_weights_data(self.name, 'kernel')

        self.add_weights_variable(name='weight', var_name='w{index}', data=data, quantizer=quantizer, compression=compression)

    def add_bias(self, quantizer=None):
        data = self.model.get_weights_data(self.name, 'bias')
        precision = None
        type_name = None
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            precision = IntegerPrecisionType(width=1, signed=False)
            type_name = 'bias{index}_t'
            quantizer = None # Don't quantize non-existant bias

        self.add_weights_variable(name='bias', var_name='b{index}', type_name=type_name, precision=precision, data=data, quantizer=quantizer)

    def add_weights_variable(self, name, var_name=None, type_name=None, precision=None, data=None, quantizer=None, compression=False):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var=name)
        elif type_name is None:
            # If precision is specified but no type name is given, assign a dedicated
            # type name made from variable name and layer index
            type_name = name + '{index}_t'

        if type_name is None:
            _, type_name = self.model.config.get_precision(self, var=name)

        if data is None:
            data = self.model.get_weights_data(self.name, name)
        elif isinstance(data, six.string_types):
            data = self.model.get_weights_data(self.name, data)

        data_unquantized = data
        exponent_type = False
        if quantizer is not None:
            precision = quantizer.hls_type
            type_name = name + '{index}_t'
            data = quantizer(data)
            if isinstance(quantizer.hls_type, ExponentPrecisionType):
                exponent_type = True

        if compression:
            #TODO reuse factor may not be available here
            var = CompressedWeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, reuse_factor=self.get_attr('reuse_factor', 1), index=self.index)
        elif exponent_type:
            var = ExponentWeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index)
        else:
            var = WeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index)

        var.data_unquantized = data_unquantized

        self.set_attr(name, var)

    def _default_function_params(self):
        params = {}
        params.update(self.attributes)
        params['config'] = 'config{}'.format(self.index)
        params['input_t'] = self.get_input_variable().type.name
        params['output_t'] = self.get_output_variable().type.name
        params['input'] = self.get_input_variable().name
        params['output'] = self.get_output_variable().name

        return params

    def _default_config_params(self):
        params = {}
        params.update(self.attributes)
        params['iotype'] = self.model.config.get_config_value('IOType')
        params['reuse'] = self.get_attr('reuse_factor')

        return params

    def get_layer_precision(self):
        precision = {}
        for data_type in self.types.values():
            precision[data_type.name] = data_type
        return precision

    def get_numbers_cpp(self):
        numbers = ''
        for k, v in self.get_output_variable().get_shape():
            numbers += '#define {} {}\n'.format(k,v)

        return numbers

    def precision_cpp(self):
        return 'typedef {precision} layer{index}_t;'.format(precision=self.get_output_variable().precision, index=self.index)

class Input(Layer):
    def initialize(self):
        shape = self.attributes['input_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_INPUT_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        if self.index == 1:
            default_type_name = 'input_t'
        else:
            default_type_name = 'input{}_t'.format(self.index)
        type_name = self.attributes.get('type_name', default_type_name)
        precision = self.attributes.get('precision', None)
        self.add_output_variable(shape, dims, var_name=self.name, type_name=type_name, precision=precision)

class Reshape(Layer):
    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        out_name = self.outputs[0]
        proxy = self.get_input_variable()
        out = InplaceVariable(shape, dims, proxy)

        self.set_attr(out_name, out)

class Dense(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),

        WeightAttribute('weight'),
        WeightAttribute('bias'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        shape = self.get_input_variable().shape[:]
        shape[-1] = self.attributes['n_out']
        if len(shape) > 1:
            dims = ['N_LAYER_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        else:
            dims = ['N_LAYER_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'), compression=self.model.config.get_compression(self))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

class Conv1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),

        Attribute('n_chan'),
        Attribute('n_filt'),

        Attribute('filt_width'),
        Attribute('stride_width'),

        Attribute('pad_left'),
        Attribute('pad_right'),

        WeightAttribute('weight'),
        WeightAttribute('bias'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]

        self.add_output_variable(shape, dims)
        self.add_weights(quantizer = self.get_attr('weight_quantizer'))
        self.add_bias(quantizer = self.get_attr('bias_quantizer'))

class SeparableConv1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),

        Attribute('n_chan'),
        Attribute('n_filt'),

        Attribute('filt_width'),
        Attribute('stride_width'),

        Attribute('pad_left'),
        Attribute('pad_right'),
        
        WeightAttribute('depthwise'),
        WeightAttribute('pointwise'),
        WeightAttribute('bias'),

        TypeAttribute('depthwise'),
        TypeAttribute('pointwise'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        
        depthwise_data = self.model.get_weights_data(self.name, 'depthwise_kernel')
        pointwise_data = self.model.get_weights_data(self.name, 'pointwise_kernel')

        self.add_weights_variable(name='depthwise', var_name='d{index}', data=depthwise_data, quantizer=self.get_attr('depthwise_quantizer'))
        self.add_weights_variable(name='pointwise', var_name='p{index}', data=pointwise_data, quantizer=self.get_attr('pointwise_quantizer'))
        
        zero_bias_data = np.zeros((self.attributes['n_chan'],))
        precision = IntegerPrecisionType(width=1, signed=False)
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data, precision=precision)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

class Conv2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),

        Attribute('out_height'),
        Attribute('out_width'),

        Attribute('n_chan'),
        Attribute('n_filt'),

        Attribute('filt_height'),
        Attribute('filt_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),

        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),

        WeightAttribute('weight'),
        WeightAttribute('bias'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

class Conv2DBatchnorm(Conv2D):
    def _get_folded_weights(self):
        """
        Function to get the batchnorm folded weights.
        This function converts the weights by folding batchnorm parameters into
        the weight of QConv2D. The high-level equation:
        W_fold = gamma * W / sqrt(variance + epsilon)
        bias_fold = gamma * (bias - moving_mean) / sqrt(variance + epsilon) + beta
        """
        kernel = self.model.get_weights_data(self.name, 'kernel')
        bias = self.model.get_weights_data(self.name, 'bias')
        if bias is None:
            bias = 0

        # get batchnorm weights and moving stats
        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')
        moving_mean = self.model.get_weights_data(self.name, 'moving_mean')
        moving_variance = self.model.get_weights_data(self.name, 'moving_variance')
        # get the inversion factor so that we replace division by multiplication
        inv = np.reciprocal(np.sqrt(moving_variance + self.get_attr('epsilon')))
        if gamma is not None:
            inv *= gamma

        # wrap conv kernel and bias with bn parameters
        folded_kernel = inv * kernel
        folded_bias = inv * (bias - moving_mean) + beta

        return [folded_kernel, folded_bias]

    def initialize(self):
        super(Conv2DBatchnorm, self).initialize()
        folded_weights, folded_bias = self._get_folded_weights()
        if self.model.config.is_resource_strategy(self) and self.model.config.backend.name in ['Vivado', 'VivadoAccelerator']:
            self.weights['weight'].data_unquantized = np.transpose(folded_weights, axes=[3, 0, 1, 2])
            self.weights['weight'].data = self.get_attr('weight_quantizer')(self.weights['weight'].data_unquantized)

        else:
            self.weights['weight'].data_unquantized = folded_weights
            self.weights['weight'].data = self.get_attr('weight_quantizer')(folded_weights)
        self.weights['bias'].data_unquantized = folded_bias
        bias_q = self.get_attr('bias_quantizer')
        if bias_q is not None:
            self.weights['bias'].data = bias_q(folded_bias)

class SeparableConv2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),

        Attribute('out_height'),
        Attribute('out_width'),

        Attribute('n_chan'),
        Attribute('n_filt'),

        Attribute('filt_height'),
        Attribute('filt_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),

        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        
        WeightAttribute('depthwise'),
        WeightAttribute('pointwise'),
        WeightAttribute('bias'),

        TypeAttribute('depthwise'),
        TypeAttribute('pointwise'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        
        depthwise_data = self.model.get_weights_data(self.name, 'depthwise_kernel')
        pointwise_data = self.model.get_weights_data(self.name, 'pointwise_kernel')

        self.add_weights_variable(name='depthwise', var_name='d{index}', data=depthwise_data, quantizer=self.get_attr('depthwise_quantizer'))
        self.add_weights_variable(name='pointwise', var_name='p{index}', data=pointwise_data, quantizer=self.get_attr('pointwise_quantizer'))
        
        zero_bias_data = np.zeros((self.attributes['n_chan'],))
        precision = IntegerPrecisionType(width=1, signed=False)
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data, precision=precision)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

class DepthwiseConv2D(Conv2D):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_chan']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_CHAN_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

        depthwise_data = self.model.get_weights_data(self.name, 'depthwise_kernel')
        self.add_weights_variable(name='weight', var_name='w{index}', data=depthwise_data, quantizer=self.get_attr('depthwise_quantizer'))

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

class Pooling1D(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('n_filt'),

        Attribute('pool_width'),
        Attribute('stride_width'),

        Attribute('pad_left'),
        Attribute('pad_right'),

        ChoiceAttribute('pool_op', ['Max', 'Average'])
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])
        self.set_attr('implementation', self.model.config.get_conv_implementation(self).lower())

class Pooling2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),

        Attribute('out_height'),
        Attribute('out_width'),

        Attribute('n_filt'),

        Attribute('pool_height'),
        Attribute('pool_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),

        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),

        ChoiceAttribute('pool_op', ['Max', 'Average'])
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])
        self.set_attr('implementation', self.model.config.get_conv_implementation(self).lower())

class GlobalPooling1D(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt'),

        ChoiceAttribute('pool_op', ['Max', 'Average'])
    ]

    def initialize(self):
        shape = [self.attributes['n_filt']]
        dims = ['N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))

class GlobalPooling2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_filt'),

        ChoiceAttribute('pool_op', ['Max', 'Average'])
    ]

    def initialize(self):
        shape = [self.attributes['n_filt']]
        dims = ['N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))

class ZeroPadding1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),
        Attribute('n_chan'),

        Attribute('pad_left'),
        Attribute('pad_right'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_chan']]
            dims = ['OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_width']]
            dims = ['N_CHAN_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims, precision=inp.type.precision)

class ZeroPadding2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_chan'),

        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_chan']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_CHAN_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims, precision=inp.type.precision)

class Activation(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('activation', value_type=str),
        #Attribute('table_size', default=1024),
        
        #TypeAttribute('table')
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        self.set_attr('n_in', self.get_input_variable().size())

class ParametrizedActivation(Activation):
    def _get_act_function_name(self):
        act = self.get_attr('activation').lower()
        if act == 'leakyrelu':
            return 'leaky_relu'
        elif act == 'thresholdedrelu':
            return 'thresholded_relu'
        else:
            return act # ELU activation

class PReLU(Activation):
    def initialize(self):
        super(PReLU, self).initialize()
        self.add_weights_variable(name='alpha', var_name='a{index}')

class Softmax(Activation):
    _expected_attributes = [
        ChoiceAttribute('implementation', ['latency', 'stable', 'legacy'], default='stable')
    ]

    def initialize(self):
        super(Softmax, self).initialize()

class TernaryTanh(Activation):
    def initialize(self):
        super(TernaryTanh, self).initialize()

class BatchNormalization(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),

        WeightAttribute('scale'),
        WeightAttribute('bias'),

        TypeAttribute('scale'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')
        mean = self.model.get_weights_data(self.name, 'moving_mean')
        var = self.model.get_weights_data(self.name, 'moving_variance')

        scale = gamma / np.sqrt(var + self.get_attr('epsilon'))
        bias = beta - gamma * mean / np.sqrt(var + self.get_attr('epsilon'))

        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias)

class Merge(Layer):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        if np.prod(inp2.shape) > np.prod(inp1.shape):
            shape = inp2.shape
            dims = inp2.dim_names
        else:
            shape = inp1.shape
            dims = inp1.dim_names
        self.add_output_variable(shape, dims)

class Dot(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        assert(inp1.shape == inp2.shape)
        if len(inp1.shape) > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')

        self.add_output_variable(shape=[1], dim_names=['OUT_DOT_{}'.format(self.index)])

class Concatenate(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        axis = self.attributes['axis']
        if axis > 0: axis -= 1
        shape = inp1.shape[:]
        shape[axis] += inp2.shape[axis]
        rank = len(shape)
        if rank > 1:
            dims = ['OUT_CONCAT_{}_{}'.format(i, self.index) for i in range(rank)]
        else:
            dims = ['OUT_CONCAT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

class BiasAdd(Merge): # TensorFlow's operator that gets merged into Dense/Conv
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        dims = inp.dim_names
        self.add_bias()
        self.add_output_variable(shape, dims)

class Resize(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        if len(inp.shape) == 2: # 1D -> width + chan
            shape = [self.get_attr('out_width'), self.get_attr('n_chan')]
            dims = ['OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        elif len(inp.shape) == 3: # 2D -> height + width + chan
            shape = [self.get_attr('out_height'), self.get_attr('out_width'), self.get_attr('n_chan')]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        self.add_output_variable(shape, dims, precision=inp.type.precision)

class Transpose(Layer):
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        perm = self.get_attr('perm')
        self.set_attr('dim', '{}d'.format(len(inp.shape)))

        if len(perm) > 3:
            raise Exception('ERROR: Transpose of tensors with rank > 3 is not yet supported.')
        
        #ONNX double transpose specific, sometimes ONNX injects
        #useless double transpose layers when converting 
        #from other frameworks
        if len(perm) == 1:
            shape = inp.shape #dummy shape
            dims = ['DUMMY'] #dummy dims
            self.set_attr('perm', [0])
        else:
            shape = [inp.shape[i] for i in perm]

        self.set_attr('perm_str', ','.join([str(i) for i in perm]))
        
        if len(shape) == 2:
            self.set_attr('perm_str', ','.join(['0'] + [str(i+1) for i in perm]))
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', 1)
            self.set_attr('height', inp.shape[0])
            self.set_attr('width', inp.shape[1])
        elif len(shape) > 2:
            dims = ['OUT_DEPTH_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', inp.shape[0])
            self.set_attr('height', inp.shape[1])
            self.set_attr('width', inp.shape[2])
        self.add_output_variable(shape, dims, precision=inp.type.precision)

class Embedding(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('vocab_size'),

        WeightAttribute('embeddings'),
        TypeAttribute('embeddings'),
    ]

    def initialize(self):
        shape = self.get_input_variable().shape[:]
        shape += [self.attributes['n_out']]
        if len(shape) > 1:
            dims = ['N_LAYER_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        else:
            dims = ['N_LAYER_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

        data = self.model.get_weights_data(self.name, 'embeddings')
        self.add_weights_variable(name='embeddings', var_name='e{index}', data=data)

class SimpleRNN(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),

        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = ['N_TIME_STEPS_{}'.format(self.index), 'N_OUT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_out']]
            dims = ['N_OUT_{}'.format(self.index)]
        
        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = ['N_OUT_{}'.format(self.index)]
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t')
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t')

        #weights
        self.add_weights()

        #recurrent weights
        recurrent_weight = self.model.get_weights_data(self.name, 'recurrent_kernel')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)

        #biases
        biases = self.model.get_weights_data(self.name , 'bias')
        self.add_weights_variable(name='bias', var_name='b{index}', data=biases)

class LSTM(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('recurrent_activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),
        Attribute('time_major', value_type=bool, default=False),

        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),
        WeightAttribute('recurrent_bias'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
        TypeAttribute('recurrent_bias'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = ['N_TIME_STEPS_{}'.format(self.index), 'N_OUT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_out']]
            dims = ['N_OUT_{}'.format(self.index)]

        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = ['N_OUT_{}'.format(self.index)]
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t')
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t')

        #weights
        self.add_weights()

        #recurrent weights
        recurrent_weight = self.model.get_weights_data(self.name, 'recurrent_kernel')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)

        #biases
        biases = self.model.get_weights_data(self.name , 'bias')
        self.add_weights_variable(name='bias', var_name='b{index}', data=biases)

        recurrent_bias = np.zeros(recurrent_weight.shape[1])
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}', data=recurrent_bias)

class GRU(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('recurrent_activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),
        Attribute('time_major', value_type=bool, default=False),
        ChoiceAttribute('apply_reset_gate', ['before', 'after'], default='after'),

        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),
        WeightAttribute('recurrent_bias'),

        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
        TypeAttribute('recurrent_bias'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = ['N_TIME_STEPS_{}'.format(self.index), 'N_OUT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_out']]
            dims = ['N_OUT_{}'.format(self.index)]

        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = ['N_OUT_{}'.format(self.index)]
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t')
            self.add_output_variable(state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t')

        #weights
        self.add_weights()

        #recurrent weights
        recurrent_weight = self.model.get_weights_data(self.name, 'recurrent_kernel')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)

        #biases array is actually a 2-dim array of arrays (bias + recurrent bias)
        #both arrays have shape: n_units * 3 (z, r, h_cand)
        biases = self.model.get_weights_data(self.name , 'bias')
        self.add_weights_variable(name='bias', var_name='b{index}', data=biases[0])
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}', data=biases[1])

class GarNet(Layer):
    ref_impl = False

    def initialize(self):
        reuse_factor = self.model.config.get_reuse_factor(self)
        if self.attributes['n_vertices'] % reuse_factor != 0:
            raise Exception('GarNet vertex loop has no bound check; number of vertices must be divisible by the reuse factor ({}).'.format(reuse_factor))

        self._initialize_transforms()

        if self.attributes['collapse']:
            shape = [self._output_features]
            dims = ['OUT_FEATURES_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_vertices'], self._output_features]
            dims = ['VERTICES_{}'.format(self.index),'OUT_FEATURES_{}'.format(self.index)]
            partition_factor = self._output_features * (self.attributes['n_vertices'] // reuse_factor)

        self.add_output_variable(shape, dims)

    def _initialize_transforms(self):
        n_propagate = self.attributes['n_propagate']
        n_aggregators = self.attributes['n_aggregators']
        n_out_features = self.attributes['n_out_features']

        if self.ref_impl:
            weights_source = [
                ('input_transform', 'FLR', 'kernel'),
                ('input_transform', 'FLR', 'bias'),
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'kernel'),
                ('output_transform', 'Fout', 'bias')
            ]

        else:
            quantize = (self.get_attr('quantizer') is not None)
            kernel, bias = self._make_input_transform_weights(n_propagate, n_aggregators, n_out_features, quantize=quantize)

            self._add_variable('input_transform_weights', 'input_transform_w{index}', kernel, frac_width=10, quantize=quantize)
            self._add_variable('input_transform_biases', 'input_transform_b{index}', bias, frac_width=10, quantize=quantize)
            #dummy
            self.add_weights_variable(name='output_transform_weights', var_name='output_transform_w{index}', data=np.ones(1))

            weights_source = [
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'bias')
            ]

        for op_name, lname, wtype in weights_source:
            data = self.model.get_weights_data(self.name, '{name}/{lname}_{wtype}:0'.format(name=self.name, lname=lname, wtype=wtype))
            if wtype == 'kernel':
                data = data.transpose((1, 0))
                vtype = 'weights'
            else:
                vtype = 'biases'

            name = '{}_{}'.format(op_name, vtype)
            var_name = '{}_{}{{index}}'.format(op_name, vtype[0])

            self._add_variable(name, var_name, data, frac_width=10, quantize=False)

        self._output_features = self.attributes['n_out_features']

    def _make_input_transform_weights(self, n_propagate, n_aggregators, n_out_features, quantize=False, sublayer=''):
        # Due to linearity of the input transform, input weights and biases can be contracted away at conversion time

        output_transform_kernel = self.model.get_weights_data(self.name, '{name}/Fout{sublayer}_kernel:0'.format(name=self.name, sublayer=sublayer)) # [(n_aggregators, n_propagate), n_out_features]
        output_transform_kernel = output_transform_kernel.reshape((n_aggregators, n_propagate, n_out_features))
        if quantize:
            output_transform_kernel = self.get_attr('quantizer')(output_transform_kernel)

        input_transform_kernel = self.model.get_weights_data(self.name, '{name}/FLR{sublayer}_kernel:0'.format(name=self.name, sublayer=sublayer)) # [n_in_features, n_propagate]
        if quantize:
            input_transform_kernel = self.get_attr('quantizer')(input_transform_kernel)
        data = np.dot(input_transform_kernel, output_transform_kernel) # [n_in_features, n_aggregators, n_out_features]
        kernel = data.transpose((2, 1, 0))

        input_transform_bias = self.model.get_weights_data(self.name, '{name}/FLR{sublayer}_bias:0'.format(name=self.name, sublayer=sublayer)) # [n_propagate]
        if quantize:
            input_transform_bias = self.get_attr('quantizer')(input_transform_bias)
        data = np.dot(input_transform_bias, output_transform_kernel) # [n_aggregators, n_out_features]
        bias = data.transpose((1, 0))

        return kernel, bias

    def _add_variable(self, name, var_name, data, frac_width=10, quantize=False):
        # Wrapper for add_weights_variable with precision determination from data

        # automatically make the variable unsigned if data are all positive
        signed = (np.amin(data) < 0.)
        
        int_width = find_minimum_width(data, signed=signed)

        if quantize:
            precision = IntegerPrecisionType(width=int_width, signed=signed)
        else:
            width = int_width + frac_width
            precision = FixedPrecisionType(width=width, integer=int_width, signed=signed, rounding_mode='AP_RND', saturation_mode='AP_SAT')
            
        self.add_weights_variable(name=name, var_name=var_name, data=data, precision=precision)
        
class GarNetStack(GarNet):
    def _initialize_transforms(self):
        self._sublayer_weights = []

        quantize = (self.get_attr('quantizer') is not None)

        for il in range(self.attributes['n_sublayers']):
            sublayer_weights = {}

            n_aggregators = self.attributes['n_aggregators'][il]
            n_out_features = self.attributes['n_out_features'][il]
            n_propagate = self.attributes['n_propagate'][il]

            kernel, bias = self._make_input_transform_weights(n_propagate, n_aggregators, n_out_features, quantize=quantize, sublayer=il)

            name = 'input_transform_{}_weights'.format(il)
            self._add_variable(name, 'input_transform_{}_w{{index}}'.format(il), kernel, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_weights'] = self.weights[name]

            name = 'input_transform_{}_biases'.format(il)
            self._add_variable(name, 'input_transform_{}_b{{index}}'.format(il), bias, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_biases'] = self.weights[name]
        
            weights_source = [
                ('aggregator_distance', 'S{}'.format(il), 'kernel'),
                ('aggregator_distance', 'S{}'.format(il), 'bias'),
                ('output_transform', 'Fout{}'.format(il), 'bias')
            ]
    
            for op_name, lname, wtype in weights_source:
                data = self.model.get_weights_data(self.name, '{name}/{lname}_{wtype}:0'.format(name=self.name, lname=lname, wtype=wtype))
                if wtype == 'kernel':
                    data = data.transpose((1, 0))
                    vtype = 'weights'
                else:
                    vtype = 'biases'

                name = '{}_{}_{}'.format(op_name, il, vtype)
                var_name = '{}_{}_{}{{index}}'.format(op_name, il, vtype[0])

                self._add_variable(name, var_name, data, frac_width=10, quantize=False)
                sublayer_weights['{}_{}'.format(op_name, vtype)] = self.weights[name]

            self._sublayer_weights.append(sublayer_weights)

        self._output_features = self.attributes['n_out_features'][-1]

layer_map = {
    'Input'                  : Input,
    'InputLayer'             : Input,
    'Activation'             : Activation,
    'QActivation'            : Activation,
    'LeakyReLU'              : ParametrizedActivation,
    'ThresholdedReLU'        : ParametrizedActivation,
    'ELU'                    : ParametrizedActivation,
    'PReLU'                  : PReLU,
    'Softmax'                : Softmax,
    'TernaryTanh'            : TernaryTanh,
    'Reshape'                : Reshape,
    'Dense'                  : Dense,
    'BinaryDense'            : Dense,
    'TernaryDense'           : Dense,
    'QDense'                 : Dense,
    'Conv1D'                 : Conv1D,
    'QConv1D'                : Conv1D,
    'Conv2D'                 : Conv2D,
    'BinaryConv2D'           : Conv2D,
    'QConv2D'                : Conv2D,
    'QConv2DBatchnorm'       : Conv2DBatchnorm,
    'SeparableConv1D'        : SeparableConv1D,
    'SeparableConv2D'        : SeparableConv2D,
    'DepthwiseConv2D'        : DepthwiseConv2D,
    'BatchNormalization'     : BatchNormalization,
    'QBatchNormalization'    : BatchNormalization,
    'MaxPooling1D'           : Pooling1D,
    'AveragePooling1D'       : Pooling1D,
    'MaxPooling2D'           : Pooling2D,
    'AveragePooling2D'       : Pooling2D,
    'GlobalMaxPooling1D'     : GlobalPooling1D,
    'GlobalAveragePooling1D' : GlobalPooling1D,
    'GlobalMaxPooling2D'     : GlobalPooling2D,
    'GlobalAveragePooling2D' : GlobalPooling2D,
    'ZeroPadding1D'          : ZeroPadding1D,
    'ZeroPadding2D'          : ZeroPadding2D,
    'Merge'                  : Merge,
    'Dot'                    : Dot,
    'Concatenate'            : Concatenate,
    'Resize'                 : Resize,
    'UpSampling1D'           : Resize,
    'UpSampling2D'           : Resize,
    'Transpose'              : Transpose,
    'Embedding'              : Embedding,
    'SimpleRNN'              : SimpleRNN,
    'LSTM'                   : LSTM,
    'GRU'                    : GRU,
    'GarNet'                 : GarNet,
    'GarNetStack'            : GarNetStack,
    # TensorFlow-specific layers:
    'BiasAdd'                : BiasAdd,
}

def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
