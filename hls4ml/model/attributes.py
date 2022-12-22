from collections.abc import MutableMapping

from hls4ml.model.types import (InplaceVariable, NamedType, Source,
                                TensorVariable, WeightVariable)
from hls4ml.utils.string_utils import convert_to_pascal_case


class Attribute:
    def __init__(self, name, value_type=int, default=None, configurable=False):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.configurable = configurable

    def validate_value(self, value):
        if self.value_type is not None:
            return issubclass(type(value), self.value_type)
        else:
            return True  # Meaning we don't care

    @property
    def config_name(self):
        return convert_to_pascal_case(self.name)


class ConfigurableAttribute(Attribute):
    def __init__(self, name, value_type=int, default=None):
        super().__init__(name, value_type, default, configurable=True)


class TypeAttribute(Attribute):
    def __init__(self, name, default=None, configurable=True):
        if not name.endswith('_t'):
            name += '_t'
        super().__init__(name, value_type=NamedType, default=default, configurable=configurable)


class ChoiceAttribute(Attribute):
    def __init__(self, name, choices, default=None, configurable=True):
        super().__init__(name, value_type=list, default=default, configurable=configurable)
        assert len(choices) > 0
        if default is not None:
            assert default in choices
        self.choices = choices
        self.value_type = str(self.choices)

    def validate_value(self, value):
        return value in self.choices


class VariableAttribute(Attribute):
    def __init__(self, name):
        super().__init__(name, value_type=WeightVariable, default=None, configurable=False)


class WeightAttribute(Attribute):
    def __init__(self, name):
        super().__init__(name, value_type=WeightVariable, default=None, configurable=False)


class CodeAttrubute(Attribute):
    def __init__(self, name):
        super(WeightAttribute, self).__init__(name, value_type=Source, default=None, configurable=False)


class AttributeDict(MutableMapping):
    def __init__(self, layer):
        self.layer = layer
        self.attributes = {}
        self._expected_attributes = [a.name for a in self.layer.expected_attributes]

    def __getitem__(self, key):
        return self.attributes[key]

    def __len__(self):
        return len(self.attributes)

    def __iter__(self):
        yield from self.attributes.keys()

    def __setitem__(self, key, value):
        if isinstance(value, (TensorVariable, InplaceVariable)):
            self.layer.model.register_output_variable(key, value)
            self.attributes['result_t'] = value.type
            if key in self._expected_attributes and key in self.layer.outputs:
                key = 'out_' + key
        elif isinstance(value, WeightVariable):
            self.attributes[key + '_t'] = value.type

        self.attributes[key] = value

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
        yield from precision_keys

    def __setitem__(self, key, value):
        self.attributes[key] = value

    def __delitem__(self, key):
        self.attributes.remove(key)


class WeightMapping(AttributeMapping):
    def __init__(self, attributes):
        super().__init__(attributes, WeightVariable)


class VariableMapping(AttributeMapping):
    def __init__(self, attributes):
        super().__init__(attributes, (TensorVariable, InplaceVariable))

    def __getitem__(self, key):
        if 'out_' + key in self.attributes:
            return self.attributes['out_' + key]
        else:
            return self.attributes[key]

    def __iter__(self):
        precision_keys = [k for k, v in self.attributes.items() if isinstance(v, self.clazz)]
        for key in precision_keys:
            if key.startswith('out_'):
                yield key[len('out_') :]
            else:
                yield key
        super().__iter__()


class TypeMapping(AttributeMapping):
    def __init__(self, attributes):
        super().__init__(attributes, NamedType)


class CodeMapping(AttributeMapping):
    def __init__(self, attributes):
        super().__init__(attributes, Source)
