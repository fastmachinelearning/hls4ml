import numpy as np
import six

from hls4ml.model.hls_types import HLSType
from hls4ml.model.hls_types import TensorVariable, WeightVariable, CompressedWeightVariable, ExponentWeightVariable, InplaceVariable
from hls4ml.model.hls_types import IntegerPrecisionType, FixedPrecisionType, ExponentPrecisionType
from hls4ml.model.hls_types import find_minimum_width

from hls4ml.model.hls_attributes import Attribute, WeightAttribute, TypeAttribute, ChoiceAttribute
from hls4ml.model.hls_attributes import AttributeDict, AttributeMapping

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

        self._function_template = self.model.config.backend.get_function_template(self.__class__.__name__)
        self._config_template = self.model.config.backend.get_config_template(self.__class__.__name__)
        self.include_list = self.model.config.backend.get_include_list(self.__class__.__name__)
        self.weights = AttributeMapping(self.attributes, WeightVariable)
        self.variables = AttributeMapping(self.attributes, TensorVariable)
        self.types = AttributeMapping(self.attributes, HLSType)

        # We set 'accum' precision to match input tensor's precision if 'accum' was not explicitly set
        def_type_obj, _ = self.model.config.get_precision(self, 'default')
        acc_type_obj, acc_type_name = self.model.config.get_precision(self, 'accum')

        inp = self.get_input_variable()
        if inp is not None:
            inp_type_obj = inp.type.precision
        else:
            inp_type_obj = def_type_obj

        if acc_type_obj == def_type_obj: # 'accum' precision not defined in config
            acc_type_obj = inp_type_obj # use input tensor's precision for 'accum'

        accum_t = HLSType(acc_type_name, acc_type_obj) 
        self.set_attr('accum_t', accum_t)

        self.reuse_factor = self.model.config.get_reuse_factor(self)

        layer_config = self.model.config.get_layer_config(self)
        for config_key, config_value in layer_config.items():
            if config_key in self.attributes:
                print('WARNING: Config parameter "{}" overwrites an existing attribute in layer "{}" ({})'.format(config_key, self.name, self.__class__.__name__))
            if config_key.endswith('_t') and isinstance(config_value, str): #TODO maybe move this to __setitem__ of AttributeDict?
                precision = self.model.config.backend.convert_precision_string(config_value)
                config_value = HLSType(self.name + config_key, precision)
            self.attributes[config_key] = config_value

        self.initialize()
        self.model.config.backend.initialize_layer(self)
        self._validate_attributes()

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
                        .format(attr_name, self.name, self.__class__.__name__, exp_attr.value_type, type(attr_value), attr_value))
            else:
                pass # TODO layer contains attribute that is not expected. we can log this for debugging
        
        # If any expected attributes remain, try adding their default values
        for attr_name, attr in all_attributes.items():
            if attr.default is not None:
                self.set_attr(attr_name, attr.default)
            else:
                raise Exception('Attribute "{}" of layer {} ({}) not set and no default value is specified.'.format(attr_name, self.name, self.__class__.__name__))

    def get_input_node(self, input_name=None):
        if input_name is not None:
            return self.model.graph.get(input_name)
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

    def add_output_variable(self, shape, dim_names, out_name=None, var_name='layer{index}_out', type_name='layer{index}_t', precision=None, pragma='auto'):
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
            var = CompressedWeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, reuse_factor=self.reuse_factor, index=self.index)
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
        params['index'] = self.index
        params['iotype'] = self.model.config.get_config_value('IOType')
        params['reuse'] = self.reuse_factor

        return params

    def get_layer_precision(self):
        precision = {}
        for data_type in self.types.values():
            precision[data_type.name] = data_type
        return precision

    # myproject.cpp/h
    def function_cpp(self):
        raise NotImplementedError

    # parameters.h
    def config_cpp(self):
        raise NotImplementedError

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

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

class Reshape(Layer):
    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        out_name = self.outputs[0]
        proxy = self.get_input_variable()
        out = InplaceVariable(shape, dims, proxy, index=self.get_input_node().index)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

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
        shape = [self.attributes['n_out']]
        dims = ['N_LAYER_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'), compression=self.model.config.get_compression(self))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['nzeros'] = self.get_weights('weight').nzeros
        params['nonzeros'] = self.get_weights('weight').nonzeros
        params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('weight').type.precision)

        return self._config_template.format(**params)

class Conv1D(Layer):
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

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        input_dims = self.get_input_variable().dim_names
        if self.get_attr('data_format') == 'channels_last':
            params['n_in'] = '*'.join([str(k) for k in input_dims[:-1]])
            params['n_chan'] = input_dims[-1]
        else:
            params['n_in'] = '*'.join([str(k) for k in input_dims[1:]])
            params['n_chan'] = input_dims[0]
        params['dilation'] = self.get_attr('dilation', 1)
        params['n_filt'] = 'N_FILT_{}'.format(self.index)
        params['out_width'] = 'N_OUTPUTS_{}'.format(self.index)
        params['nzeros'] = self.get_weights('weight').nzeros

        params['config_t'] = 'config{}_mult'.format(self.index)
        conv_config = self._config_template[0].format(**params)

        mult_params = self._default_config_params()
        mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_width')
        mult_params['n_out'] = self.get_attr('n_filt')
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('weight').type.precision)
        mult_config = self._config_template[1].format(**mult_params)

        return mult_config + '\n' + conv_config

class SeparableConv1D(Layer):
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
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = self.get_weights('depthwise').name
        params['p'] = self.get_weights('pointwise').name
        params['b'] = self.get_weights('bias').name
        params['z'] = self.get_weights('zero_bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        # Separable master config
        params = {}
        params['index'] = self.index
        params['depthwise_config'] = 'config{}_depthwise'.format(self.index)
        params['pointwise_config'] = 'config{}_pointwise'.format(self.index)
        sep_config = self._config_template[0].format(**params)

        # Depthwise config
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_width'] = self.get_input_variable().dim_names[0]
            params['n_chan'] = self.get_input_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[0]
            params['n_filt'] = self.get_input_variable().dim_names[1]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_filt'] = self.get_input_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]

        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('depthwise').nzeros
        params['index'] = str(self.index) + '_depthwise'
        params['weight_t'] = self.get_weights('depthwise').type

        params['config_t'] = 'config{}_depthwise_mult'.format(self.index)
        depthwise_config = self._config_template[1].format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params()
        mult_params['index'] = str(self.index) + '_depthwise'
        mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_width')
        mult_params['n_out'] = self.get_attr('n_chan')
        mult_params['weight_t'] = self.get_weights('depthwise').type
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('depthwise').type.precision)
        depthwise_mult_config = self._config_template[3].format(**mult_params)

        # Pointwise config
        params = self._default_config_params()
        input_dims = self.get_input_variable().dim_names
        if self.get_attr('data_format') == 'channels_last':
            params['in_width'] = '*'.join([str(k) for k in input_dims[:-1]])
            params['n_chan'] = input_dims[-1]
        else:
            params['in_width'] = '*'.join([str(k) for k in input_dims[1:]])
            params['n_chan'] = input_dims[0]
        
        params['filt_width'] = 1
        params['dilation'] = self.get_attr('dilation', 1)
        params['n_filt'] = 'N_FILT_{}'.format(self.index)
        params['out_width'] = 'N_OUTPUTS_{}'.format(self.index)
        params['nzeros'] = self.get_weights('pointwise').nzeros
        params['index'] = str(self.index) + '_pointwise'
        params['weight_t'] = self.get_weights('pointwise').type
        params['min_width'] = params['in_width']
        params['instructions'] = '0'

        params['config_t'] = 'config{}_pointwise_mult'.format(self.index)
        pointwise_config = self._config_template[2].format(**params)

        # Pointwise mult config
        mult_params = self._default_config_params()
        mult_params['index'] = str(self.index) + '_pointwise'
        mult_params['n_in'] = self.get_attr('n_chan')
        mult_params['n_out'] = self.get_attr('n_filt')
        mult_params['weight_t'] = self.get_weights('pointwise').type
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('pointwise').type.precision)
        pointwise_mult_config = self._config_template[4].format(**mult_params)

        return depthwise_mult_config + '\n' + depthwise_config + '\n' + pointwise_mult_config + '\n' + pointwise_config + '\n' + sep_config

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

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[2]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['n_filt'] = self.get_output_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]
        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('weight').nzeros

        params['config_t'] = 'config{}_mult'.format(self.index)
        conv_config = self._config_template[0].format(**params)

        mult_params = self._default_config_params()
        mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_height') * self.get_attr('filt_width')
        mult_params['n_out'] = self.get_attr('n_filt')
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('weight').type.precision)
        mult_config = self._config_template[1].format(**mult_params)

        return mult_config + '\n' + conv_config


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
        if self.model.config.is_resource_strategy(self) and self.model.config.backend.name == 'Vivado':
            self.weights['weight'].data_unquantized = np.transpose(folded_weights, axes=[3, 0, 1, 2])
            self.weights['weight'].data = self.get_attr('weight_quantizer')(self.weights['weight'].data_unquantized)

        else:
            self.weights['weight'].data_unquantized = folded_weights
            self.weights['weight'].data = self.get_attr('weight_quantizer')(folded_weights)
        self.weights['bias'].data_unquantized = folded_bias
        bias_q = self.get_attr('bias_quantizer')
        if bias_q is not None:
            self.weights['bias'].data = bias_q(folded_bias)

    def function_cpp(self):
        return super(Conv2DBatchnorm, self).function_cpp()

    def config_cpp(self):
        return super(Conv2DBatchnorm, self).config_cpp()

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
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = self.get_weights('depthwise').name
        params['p'] = self.get_weights('pointwise').name
        params['b'] = self.get_weights('bias').name
        params['z'] = self.get_weights('zero_bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        # Separable master config
        params = {}
        params['index'] = self.index
        params['depthwise_config'] = 'config{}_depthwise'.format(self.index)
        params['pointwise_config'] = 'config{}_pointwise'.format(self.index)
        sep_config = self._config_template[0].format(**params)

        # Depthwise config
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_input_variable().dim_names[2]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['n_filt'] = self.get_input_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]

        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('depthwise').nzeros
        params['index'] = str(self.index) + '_depthwise'
        params['weight_t'] = self.get_weights('depthwise').type

        params['config_t'] = 'config{}_depthwise_mult'.format(self.index)
        depthwise_config = self._config_template[1].format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params()
        mult_params['index'] = str(self.index) + '_depthwise'
        mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_height') * self.get_attr('filt_width')
        mult_params['n_out'] = self.get_attr('n_chan')
        mult_params['weight_t'] = self.get_weights('depthwise').type
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('depthwise').type.precision)
        depthwise_mult_config = self._config_template[3].format(**mult_params)

        # Pointwise config
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_output_variable().dim_names[0]
            params['in_width'] = self.get_output_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[2]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_output_variable().dim_names[1]
            params['in_width'] = self.get_output_variable().dim_names[2]
            params['n_filt'] = self.get_output_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]

        params['filt_height'] = params['filt_width'] = 1
        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('pointwise').nzeros
        params['index'] = str(self.index) + '_pointwise'
        params['weight_t'] = self.get_weights('pointwise').type
        params['min_height'] = params['in_height']
        params['min_width'] = params['in_width']
        params['instructions'] = '0'

        params['config_t'] = 'config{}_pointwise_mult'.format(self.index)
        pointwise_config = self._config_template[2].format(**params)

        # Pointwise mult config
        mult_params = self._default_config_params()
        mult_params['index'] = str(self.index) + '_pointwise'
        mult_params['n_in'] = self.get_attr('n_chan')
        mult_params['n_out'] = self.get_attr('n_filt')
        mult_params['weight_t'] = self.get_weights('pointwise').type
        mult_params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('pointwise').type.precision)
        pointwise_mult_config = self._config_template[4].format(**mult_params)

        return depthwise_mult_config + '\n' + depthwise_config + '\n' + pointwise_mult_config + '\n' + pointwise_config + '\n' + sep_config

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
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['n_in'] = self.get_input_variable().dim_names[0]
            params['n_out'] = self.get_output_variable().dim_names[0]
            params['n_filt'] = self.get_output_variable().dim_names[1]
        else:
            params['n_in'] = self.get_input_variable().dim_names[1]
            params['n_out'] = self.get_input_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[0]

        return self._config_template.format(**params)

class Pooling2D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[2]
        else:
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['n_filt'] = self.get_output_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]

        return self._config_template.format(**params)

class GlobalPooling1D(Layer):
    def initialize(self):
        shape = [self.attributes['n_out'], self.attributes['n_filt']]
        dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)

class GlobalPooling2D(Layer):
    def initialize(self):
        shape = [self.attributes['n_filt']]
        dims = ['N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
        else:
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]

        return self._config_template.format(**params)

class ZeroPadding1D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_chan']]
            dims = ['OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_width']]
            dims = ['N_CHAN_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_width'] = self.get_input_variable().dim_names[0]
            params['n_chan'] = self.get_input_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[0]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[1]

        return self._config_template.format(**params)

class ZeroPadding2D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_chan']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_CHAN_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]

        return self._config_template.format(**params)

class Activation(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('activation', value_type=str),
        Attribute('table_size', default=1024),
        
        TypeAttribute('table')
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        self.set_attr('n_in', self.get_input_variable().size())

        if 'table_t' not in self.attributes:
            self.set_attr('table_t', HLSType(name=self.name + '_table_t', precision=FixedPrecisionType(width=18, integer=8)))
        #if 'table_size' not in self.attributes:
        #    self.set_attr('table_size', 1024)

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['type'] = self.get_attr('activation')

        return self._config_template.format(**params)

class ParametrizedActivation(Activation):
    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self._get_act_function_name()
        params['param'] = self.get_attr('activ_param', 1.0)
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

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

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['param'] = self.get_weights('alpha').name
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

class Softmax(Activation):
    _expected_attributes = [
        ChoiceAttribute('implementation', ['latency', 'stable', 'legacy'], default='stable')
    ]

    def initialize(self):
        super(Softmax, self).initialize()

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

    def function_cpp(self):
        params = self._default_function_params()
        params['scale'] = self.get_weights('scale').name
        params['bias'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['product_type'] = self.model.config.backend.product_type(self.get_input_variable().type.precision, self.get_weights('scale').type.precision)

        return self._config_template.format(**params)

class Merge(Layer):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        shape = inp1.shape
        assert(inp1.shape == inp2.shape)
        dims = inp1.dim_names
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = {}
        params['merge'] = self.get_attr('op').lower()
        params['config'] = 'config{}'.format(self.index)
        params['input1_t'] = self.get_input_variable(self.inputs[0]).type.name
        params['input2_t'] = self.get_input_variable(self.inputs[1]).type.name
        params['output_t'] = self.get_output_variable().type.name
        params['input1'] = self.get_input_variable(self.inputs[0]).name
        params['input2'] = self.get_input_variable(self.inputs[1]).name
        params['output'] = self.get_output_variable().name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_elem'] = self.get_input_variable(self.inputs[0]).size_cpp()

        return self._config_template.format(**params)

class Dot(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        assert(inp1.shape == inp2.shape)
        if len(inp1.shape) > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')

        self.add_output_variable(shape=[1], dim_names=['OUT_DOT_{}'.format(self.index)])

    def config_cpp(self):
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        params = self._default_config_params()
        params['n_out'] = 1
        params['n_in'] = inp1.shape[0]
        params['product_type'] = self.model.config.backend.product_type(inp1.type.precision, inp2.type.precision)
        return self._config_template.format(**params)

class Concatenate(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        axis = self.attributes['axis']
        shape = inp1.shape[:]
        shape[axis] += inp2.shape[axis]
        rank = len(shape)
        if rank > 1:
            dims = ['OUT_CONCAT_{}_{}'.format(i, self.index) for i in range(rank)]
        else:
            dims = ['OUT_CONCAT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def config_cpp(self):
        params = self._default_config_params()
        for i in range(3):
            params.setdefault('n_elem1_{}'.format(i), 0)
            params.setdefault('n_elem2_{}'.format(i), 0)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        for i, (s1, s2) in enumerate(zip(inp1.shape, inp2.shape)):
            params['n_elem1_{}'.format(i)] = s1
            params['n_elem2_{}'.format(i)] = s2

        return self._config_template.format(**params)

class BiasAdd(Merge): # TensorFlow's operator that gets merged into Dense/Conv
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        dims = inp.dim_names
        self.add_bias()
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

    def config_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

class Resize(Layer):
    def initialize(self):
        shape = [self.get_attr('out_height'), self.get_attr('out_width'), self.get_attr('n_chan')]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['algorithm'] = self.get_attr('algorithm')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class Transpose(Layer):
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        perm = self.get_attr('perm')
        self.set_attr('dim', '{}d'.format(len(inp.shape)))
        if len(perm) == 4 and perm[0] == 0:
            perm = [i - 1 for i in perm[1:]]
        shape = [inp.shape[i] for i in perm]
        self.set_attr('perm_str', ','.join([str(i) for i in perm]))
        if len(shape) == 2:
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', 1)
            self.set_attr('height', shape[0])
            self.set_attr('width', shape[1])
        else:
            dims = ['OUT_DEPTH_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', shape[0])
            self.set_attr('height', shape[1])
            self.set_attr('width', shape[2])
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['dim'] = self.get_attr('dim')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class GarNet(Layer):
    ref_impl = False

    def initialize(self):
        reuse_factor = self.model.config.get_reuse_factor(self)
        if self.attributes['n_vertices'] % reuse_factor != 0:
            raise Exception('GarNet vertex loop has no bound check; number of vertices must be divisible by the reuse factor ({}).'.format(reuse_factor))

        self._initialize_transforms()

        # A bit controvertial but we are going to reshape the input variable here
        input_array = self.get_input_variable(self.inputs[0])
        partition_factor = input_array.shape[1] * (input_array.shape[0] // reuse_factor)
        input_array.pragma = ('partition', 'cyclic', partition_factor)
        
        if self.attributes['collapse']:
            shape = [self._output_features]
            dims = ['OUT_FEATURES_{}'.format(self.index)]
            pragma = 'partition'
        else:
            shape = [self.attributes['n_vertices'], self._output_features]
            dims = ['VERTICES_{}'.format(self.index),'OUT_FEATURES_{}'.format(self.index)]
            partition_factor = self._output_features * (self.attributes['n_vertices'] // reuse_factor)
            pragma = ('partition', 'cyclic' , partition_factor)

        self.add_output_variable(shape, dims, pragma=pragma)

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
        
    def function_cpp(self):
        params = self._default_function_params()

        data = self.get_input_variable(self.inputs[0])
        integer_input = self.get_input_variable(self.inputs[1])
        params['input_t'] = data.type.name
        params['input'] = data.name

        params['integer_input_t'] = integer_input.type.name
        params['nvtx'] = integer_input.name

        if self.ref_impl:
            params['impl'] = '_ref'
        else:
            params['impl'] = ''

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        params['n_vertices'] = self.attributes['n_vertices']
        params['n_vertices_width'] = int(np.log2(params['n_vertices']))
        params['distance_width'] = 12
        params['distance_nint'] = min(4, params['distance_width'] - 6) # this is tuned
        params['log2_reuse'] = int(np.log2(params['reuse']))

        ## Define default precisions for various internal arrays (can be overridden from the config file)
        # We always give 10 digits for the subintegral part
        fwidth = 10
        # Integral precision for aggr_t depends on how large the temporary sum for weighed feature mean will be
        aggr_intw = max(params['log2_reuse'], params['n_vertices_width'] - params['log2_reuse']) + 3 # safety factor 2**3
        aggr_w = aggr_intw + fwidth
        # edge_weight_aggr_t does not need the safety factor
        ew_aggr_intw = aggr_intw - 3
        ew_aggr_w = ew_aggr_intw + fwidth
        # Integral precision for norm is fixed to 4
        norm_intw = 4
        norm_w = norm_intw + fwidth

        vspecs = [
            ('edge_weight', FixedPrecisionType(10, 0, signed=False)),
            ('edge_weight_aggr', FixedPrecisionType(ew_aggr_w, ew_aggr_intw, signed=False)),
            ('aggr', FixedPrecisionType(aggr_w, aggr_intw)),
            ('norm', FixedPrecisionType(norm_w, norm_intw, signed=False))
        ]
        for vname, default_precision in vspecs:
            params['{}_t'.format(vname)], type_name = self.model.config.get_precision(self, var=vname)
            if type_name.endswith('default_t'):
                params['{}_t'.format(vname)] = str(default_precision)

        params['output_t'] = self.get_output_variable().type.name

        if self.attributes['collapse'] in ['mean', 'max']:
            params['collapse_type'] = 'collapse_{}'.format(self.attributes['collapse'])
        else:
            params['collapse_type'] = 'no_collapse'

        params['mean_by_nvert'] = str(self.attributes['mean_by_nvert']).lower()

        self._get_transforms_config(params)

        return self._config_template.format(**params)

    def _get_transforms_config(self, params):
        params['n_in_features'] = self.attributes['n_in_features']
        params['n_propagate'] = self.attributes['n_propagate']
        params['n_aggregators'] = self.get_weights('aggregator_distance_biases').shape[0]
        params['n_out_features'] = self.get_weights('output_transform_biases').shape[0]

        for wname, weights in self.weights.items():
            params[wname] = weights.name
            params['{}_t'.format(wname)] = weights.type.name
            params['{}_size'.format(wname)] = weights.data_length


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

    def _get_transforms_config(self, params):
        base_template, sublayer_template = self._config_template
        self._config_template = base_template

        params['n_sublayers'] = self.attributes['n_sublayers']
        params['n_in_features'] = self.attributes['n_in_features'][0]
        params['n_out_features'] = self.attributes['n_out_features'][-1]

        sublayer_configs = []
        for il in range(self.attributes['n_sublayers'] - 1, -1, -1):
            sub_params = {'index': self.index, 'il': il}

            for p in ['n_in_features', 'n_propagate', 'n_aggregators', 'n_out_features']:
                sub_params[p] = self.attributes[p][il]

            for wname, weights in self._sublayer_weights[il].items():
                sub_params[wname] = weights.name
                sub_params['{}_t'.format(wname)] = weights.type.name
                sub_params['{}_size'.format(wname)] = weights.data_length

            if il != self.attributes['n_sublayers'] - 1:
                sub_params['next'] = il + 1
            else:
                sub_params['next'] = 0

            sublayer_configs.append(sublayer_template.format(**sub_params))

        params['sublayer_configs'] = '\n'.join(sublayer_configs)

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
    'UpSampling2D'           : Resize,
    'Transpose'              : Transpose,
    'GarNet'                 : GarNet,
    'GarNetStack'            : GarNetStack,
    # TensorFlow-specific layers:
    'BiasAdd'                : BiasAdd,
}

def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
