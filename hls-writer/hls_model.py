from __future__ import print_function
import six
import re
import numpy as np
from enum import Enum
from collections import OrderedDict

from templates import get_config_template, get_function_template

class HLSConfig(object):
    def __init__(self, config):
        self.config = config

        self.model_precision = {}
        self.layer_type_precision = {}
        self.layer_name_precision = {}

        self.model_rf = None
        self.layer_type_rf = {}
        self.layer_name_rf = {}

        self._parse_hls_config()

    def get_config_value(self, key):
        return self.config[key]

    def get_project_name(self):
        return self.get_config_value('ProjectName')

    def get_output_dir(self):
        return self.get_config_value('OutputDir')

    def get_precision(self, layer, var='default'):
        precision = self.layer_name_precision.get(layer.name.lower() + '_' + var)
        type_name = layer.name.lower() + '_' + var + '_t'
        if precision is None:
            precision = self.layer_name_precision.get(layer.name.lower() + '_default')
            type_name = layer.name.lower() + '_default_t'

        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_' + var)
            type_name = layer.__class__.__name__ + '_' + var + '_t'
        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_default')
            type_name = layer.__class__.__name__ + '_default_t'

        if precision is None:
            precision = self.model_precision.get(var)
            type_name = var + '_default_t'
        if precision is None:
            precision = self.model_precision.get('default')
            type_name = 'model_default_t'

        if precision is None:
            raise Exception('No precision for {}->{} found and no default specified.'.format(layer.name, var))

        return (precision, type_name)

    def get_reuse_factor(self, layer):
        rf = self.layer_name_rf.get(layer.name.lower())
        if rf is None:
            rf = self.layer_type_rf.get(layer.__class__.__name__.lower())
        if rf is None:
            rf = self.model_rf

        if rf is None:
            raise Exception('No reuse factor for {} found and no default specified.'.format(layer.name))

        return rf

    def _parse_hls_config(self):
        hls_config = self.config['HLSConfig']
        model_cfg = hls_config.get('Model')
        if model_cfg is not None:
            precision_cfg = model_cfg.get('Precision')
            if precision_cfg is not None:
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.model_precision[var] = precision
                else:
                    self.model_precision['default'] = precision_cfg # Default precision for everything

            self.model_rf = model_cfg.get('ReuseFactor')

        layer_type_cfg = hls_config.get('LayerType')
        if layer_type_cfg is not None:
            for layer_type, layer_cfg in layer_type_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_type_precision[layer_type.lower() + '_' + var] = precision
                else:
                    self.layer_type_precision[layer_type.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_type_rf[layer_type.lower()] = rf

        layer_name_cfg = hls_config.get('LayerName')
        if layer_name_cfg is not None:
            for layer_name, layer_cfg in layer_name_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_name_precision[layer_name.lower() + '_' + var] = precision
                else:
                    self.layer_name_precision[layer_name.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_name_rf[layer_name.lower()] = rf

class HLSModel(object):
    def __init__(self, config, data_reader, layer_list, inputs=None, outputs=None):
        self.config = HLSConfig(config)
        self.reader = data_reader

        # If not provided, assumes layer_list[0] is input, and layer_list[-1] is output
        self.inputs = inputs if inputs is not None else [layer_list[0]['name']]
        self.outputs = outputs if outputs is not None else [layer_list[-1]['name']]

        self.index = 0
        self.graph = OrderedDict()
        self.output_vars = {}

        self._make_graph(layer_list)

    def _make_graph(self, layer_list):
        for layer in layer_list:
            kind = layer['class_name']
            name = layer['name']
            inputs = layer.get('inputs', [])
            outputs = layer.get('outputs', [])
            if len(inputs) == 0:
                inputs = [next(reversed(self.graph), 'input')]
            if len(outputs) == 0:
                outputs = [name]

            node = layer_map[kind](self, name, layer, inputs, outputs)
            self.graph[name] = node

    def get_weights_data(self, layer_name, var_name):
        return self.reader.get_weights_data(layer_name, var_name)

    def quantize_data(self, data, quantize):
        ones = np.ones_like(data)
        quant_data = data
        if quantize == 2:
            quant_data = np.where(data > 0, ones, -ones)
        elif quantize == 3:
            zeros = np.zeros_like(data)
            quant_data = np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))
        return quant_data

    def next_layer(self):
        self.index += 1
        return self.index

    def get_layers(self):
        return self.graph.values()

    def get_input_variables(self):
        variables = []
        for inp in self.inputs:
            variables.append(self.graph[inp].get_output_variable())
        return variables

    def register_output_variable(self, out_name, variable):
        if out_name in self.outputs:
            variable.type = 'result_t'
        self.output_vars[out_name] = variable

    def get_output_variables(self):
        variables = []
        for out in self.outputs:
            variables.append(self.output_vars[out])
        return variables

    def get_layer_output_variable(self, output_name):
        return self.output_vars[output_name]

class Variable(object):
    def __init__(self, var_name, type_name, precision, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = type_name.format(**kwargs)
        self.precision = precision
        self.cppname = re.sub(r'\W|^(?=\d)','_', self.name)

class ArrayVariable(Variable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, pragma='partition', **kwargs):
        super(ArrayVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.shape = shape
        self.dim_names = dim_names

        if pragma == 'partition':
            self.partition()
        elif pragma == 'reshape':
            self.reshape()
        elif pragma == 'stream':
            self.stream()
        else:
            self.pragma = None

    def partition(self, type='complete', factor=None, dim=0):
        if factor:
            pragma = '#pragma HLS ARRAY_PARTITION variable={name} {type} factor={factor} dim={dim}'
        else:
            pragma = '#pragma HLS ARRAY_PARTITION variable={name} {type} dim={dim}'

        self.pragma = pragma.format(name=self.name, type=type, factor=factor, dim=dim)

    def reshape(self, type='complete', factor=None, dim=0):
        if factor:
            pragma = '#pragma HLS ARRAY_RESHAPE variable={name} {type} factor={factor} dim={dim}'
        else:
            pragma = '#pragma HLS ARRAY_RESHAPE variable={name} {type} dim={dim}'

        self.pragma = pragma.format(name=self.name, type=type, factor=factor, dim=dim)

    def stream(self, depth=1, dim=1):
        pragma = '#pragma HLS STREAM variable={name} depth={depth} dim={dim}'
        self.pragma = pragma.format(name=self.name, depth=depth, dim=dim)

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        array_shape = '*'.join([str(k) for k in self.dim_names])
        return '{type} {name}[{shape}]'.format(type=self.type, name=self.cppname, shape=array_shape)

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class WeightVariable(Variable):
    def __init__(self, var_name, type_name, precision, data=None, **kwargs):
        super(WeightVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.data = data
        self.nzeros = -1
        self.shape = None
        if self.data is not None:
            self.shape = list(self.data.shape)
            self.nzeros = 0
            for x in np.nditer(self.data, order='C'):
                if x == 0:
                    self.nzeros += 1

class Layer(object):
    def __init__(self, model, name, attributes, inputs, outputs=None):
        self.model = model
        self.name = name
        self.index = model.next_layer()
        self.inputs = inputs
        self.outputs = outputs
        if self.outputs is None:
            self.outputs = [self.name]

        self.attributes = attributes
        self._function_template = get_function_template(self.__class__.__name__)
        self._config_template = get_config_template(self.__class__.__name__)
        self.weights = OrderedDict()
        self.variables = OrderedDict()
        self.precision = OrderedDict()
        accum_precision = self.model.config.get_precision(self, 'accum')
        self.precision[accum_precision[1]] = accum_precision[0]
        self.set_attr('accum_t', accum_precision[1])

        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def set_attr(self, key, value):
        self.attributes[key] = value

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def get_input_variable(self, input_name=None):
        if input_name is not None:
            return self.model.get_layer_output_variable(input_name)
        else:
            return self.model.get_layer_output_variable(self.inputs[0])

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

        if pragma == 'auto':
            if self.model.config.get_config_value('IOType') == 'io_serial':
                pragma = 'stream'
            else:
                if self.name in self.model.inputs:
                    pragma = 'reshape'
                else:
                    pragma = 'partition'

        out = ArrayVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, pragma=pragma, index=self.index)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

        self.precision[out.type] = out.precision

    def add_weights(self, quantize=0):
        data = self.model.get_weights_data(self.name, 'kernel')

        self.add_weights_variable(name='weight', var_name='w{index}', data=data, quantize=quantize)

    def add_bias(self, quantize=0):
        data = self.model.get_weights_data(self.name, 'bias')
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            quantize = 0 # Don't quantize non-existant bias

        self.add_weights_variable(name='bias', var_name='b{index}', data=data, quantize=quantize)

    def add_weights_variable(self, name, var_name=None, type_name=None, precision=None, data=None, quantize=0):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, new_type_name = self.model.config.get_precision(self, var=name)

        if type_name is None:
            type_name = new_type_name

        if data is None:
            data = self.model.get_weights_data(self.name, name)
        elif isinstance(data, six.string_types):
            data = self.model.get_weights_data(self.name, data)

        if quantize > 0:
            data = self.model.quantize_data(data, quantize)

        var = WeightVariable(var_name, type_name=type_name, precision=precision, data=data, index=self.index)

        self.weights[name] = var
        self.precision[var.type] = var.precision

    def _default_function_params(self):
        params = {}
        params['config'] = 'config{}'.format(self.index)
        params['input_t'] = self.get_input_variable().type
        params['output_t'] = self.get_output_variable().type
        params['input'] = self.get_input_variable().name
        params['output'] = self.get_output_variable().name

        return params

    def _default_config_params(self):
        params = {}
        params.update(self.attributes)
        params['index'] = self.index
        params['iotype'] = self.model.config.get_config_value('IOType')
        params['reuse'] = self.model.config.get_reuse_factor(self)

        # data types
        for weight_name, variable in self.weights.items():
            params[weight_name + '_t'] = variable.type

        return params

    def get_layer_precision(self):
        return self.precision

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
        self.add_output_variable(shape, dims, var_name=self.name, type_name='input_t')

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

class Dense(Layer):
    def initialize(self):
        shape = [self.attributes['n_out']]
        dims = ['N_LAYER_{}'.format(self.index)]
        quantize = self.get_attr('quantize', default=0)
        self.add_output_variable(shape, dims)
        self.add_weights(quantize=quantize)
        self.add_bias(quantize=quantize)

    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()
        params['nzeros'] = self.get_weights('weight').nzeros

        return self._config_template.format(**params)

class Conv1D(Layer):
    def initialize(self):
        shape = [self.attributes['y_out'], self.attributes['n_filt']]
        dims = ['Y_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()

    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['y_in'] = self.get_input_variable().dim_names[0]
        params['n_chan'] = self.get_input_variable().dim_names[1]
        params['n_filt'] = 'N_FILT_{}'.format(self.index)
        params['y_out'] = 'Y_OUTPUTS_{}'.format(self.index)
        params['nzeros'] = self.get_weights('weight').nzeros

        return self._config_template.format(**params)

class Conv2D(Layer):
    def initialize(self):
        shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights()
        self.add_bias()

    def function_cpp(self):
        params = self._default_function_params()
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['in_height'] = self.get_input_variable().dim_names[0]
        params['in_width'] = self.get_input_variable().dim_names[1]
        params['n_chan'] = self.get_input_variable().dim_names[2]
        params['out_height'] = self.get_output_variable().dim_names[0]
        params['out_width'] = self.get_output_variable().dim_names[1]
        params['n_filt'] = self.get_output_variable().dim_names[2]
        params['nzeros'] = self.get_weights('weight').nzeros

        return self._config_template.format(**params)

class Pooling1D(Layer):
    def initialize(self):
        shape = [self.attributes['n_out'], self.attributes['n_filt']]
        dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()

        return self._config_template.format(**params)

class Pooling2D(Layer):
    def initialize(self):
        shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().dim_names[0]
        params['in_width'] = self.get_input_variable().dim_names[1]
        params['out_height'] = self.get_output_variable().dim_names[0]
        params['out_width'] = self.get_output_variable().dim_names[1]
        params['n_filt'] = self.get_output_variable().dim_names[2]

        return self._config_template.format(**params)

class Activation(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation')
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['type'] = self.get_attr('activation')
        params['n_in'] = self.get_input_variable().size_cpp()

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

class BatchNormalization(Layer):
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
        params['input1_t'] = self.get_input_variable(self.inputs[0]).type
        params['input2_t'] = self.get_input_variable(self.inputs[1]).type
        params['output_t'] = self.get_output_variable().type
        params['input1'] = self.get_input_variable(self.inputs[0]).name
        params['input2'] = self.get_input_variable(self.inputs[1]).name
        params['output'] = self.get_output_variable().name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_elem'] = self.get_input_variable(self.inputs[0]).size_cpp()

        return self._config_template.format(**params)

class Concatenate(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        shape = [sum(x) for x in zip(inp1.shape, inp2.shape)]
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

layer_map = {
    'InputLayer'         : Input,
    'Activation'         : Activation,
    'LeakyReLU'          : ParametrizedActivation,
    'ThresholdedReLU'    : ParametrizedActivation,
    'ELU'                : ParametrizedActivation,
    'PReLU'              : PReLU,
    'Dense'              : Dense,
    'BinaryDense'        : Dense,
    'TernaryDense'       : Dense,
    'Conv1D'             : Conv1D,
    'Conv2D'             : Conv2D,
    'BatchNormalization' : BatchNormalization,
    'MaxPooling1D'       : Pooling1D,
    'AveragePooling1D'   : Pooling1D,
    'MaxPooling2D'       : Pooling2D,
    'AveragePooling2D'   : Pooling2D,
    'Merge'              : Merge,
    'Concatenate'        : Concatenate,
}
