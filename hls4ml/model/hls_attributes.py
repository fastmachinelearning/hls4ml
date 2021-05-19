from collections.abc import MutableMapping

from hls4ml.model.hls_types import HLSType, TensorVariable, WeightVariable

class Attribute(object):
    def __init__(self, name, value_type=int, default=None, configurable=False):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.configurable = configurable

    def validate_value(self, value):
        if self.value_type is not None:
            return type(value) == self.value_type
        else:
            return True # Meaning we don't care

class TypeAttribute(Attribute):
    def __init__(self, name, default=None, configurable=True):
        if not name.endswith('_t'):
            name += '_t'
        super(TypeAttribute, self).__init__(name, value_type=HLSType, default=default, configurable=configurable)

class ChoiceAttribute(Attribute):
    def __init__(self, name, choices, default=None, configurable=True):
        super(ChoiceAttribute, self).__init__(name, value_type=list, default=default, configurable=configurable)
        assert(len(choices) > 0)
        if default is not None:
            assert(default in choices)
        self.choices = choices

    def validate_value(self, value):
        return value in self.choices

class VariableAttribute(Attribute):
    def __init__(self, name):
        super(VariableAttribute, self).__init__(name, value_type=WeightVariable, default=None, configurable=False)

class WeightAttribute(Attribute):
    def __init__(self, name):
        super(WeightAttribute, self).__init__(name, value_type=WeightVariable, default=None, configurable=False)

class AttributeDict(MutableMapping):
    def __init__(self, layer):
        self.layer = layer
        self.attributes = {}

    def __getitem__(self, key):
        return self.attributes[key]

    def __len__(self):
        return len(self.attributes)

    def __iter__(self):
        for key in self.attributes.keys():
            yield key

    def __setitem__(self, key, value):
        self.attributes[key] = value
        if isinstance(value, TensorVariable):
            self.layer.model.register_output_variable(key, value)
            self.attributes['result_t'] = value.type
        elif isinstance(value, WeightVariable):
            self.attributes[key + '_t'] = value.type

    def __delitem__(self, key):
        self.attributes.remove(key)

class AttributeMapping(MutableMapping):
    def __init__(self, attributes, clazz):
        self.attributes = attributes
        self.clazz = clazz

    def __getitem__(self, key):
        return self.attributes[key]

    def __len__(self):
        return sum(map(lambda x: isinstance(x, self.clazz), self.attributes.values()))

    def __iter__(self):
        precision_keys = [k for k, v in self.attributes.items() if isinstance(v, self.clazz)]
        for key in precision_keys:
            yield key

    def __setitem__(self, key, value):
        self.attributes[key] = value

    def __delitem__(self, key):
        self.attributes.remove(key)