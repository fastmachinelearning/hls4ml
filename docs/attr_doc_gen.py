import numbers

import hls4ml.backends as backends
import hls4ml.model.attributes as attributes
import hls4ml.model.layers as layers

all_backends = backends.get_available_backends()
# Removing duplicates but preserving order
all_layers = list(dict.fromkeys(layers.layer_map.values()))


class AttrList:
    def __init__(self, cls_name, cls_attrs) -> None:
        self.cls_name = cls_name
        self.config_attrs = [attr for attr in cls_attrs if attr.configurable == True]
        self.type_attrs = [attr for attr in cls_attrs if attr.__class__.__name__ == 'TypeAttribute']
        self.weight_attrs = [attr for attr in cls_attrs if attr.__class__.__name__ == 'WeightAttribute']
        self.base_attrs = [attr for attr in cls_attrs if attr not in self.config_attrs + self.type_attrs + self.weight_attrs]
        self.backend_attrs = {}

    def add_backend_attrs(self, backend_name, backend_attrs):
        self.backend_attrs[backend_name] = backend_attrs


attr_map = []

for layer_cls in all_layers:
    base_attrs = layer_cls.expected_attributes

    attr_list = AttrList(layer_cls.__name__, base_attrs)

    for backend_name in all_backends:
        backend = backends.get_backend(backend_name)

        backend_cls = backend.create_layer_class(layer_cls)
        backend_attrs = backend_cls.expected_attributes

        diff_atts = [attr for attr in backend_attrs if attr not in base_attrs]  # Sets are faster, but don't preserve order
        if len(diff_atts) > 0:
            attr_list.add_backend_attrs(backend.name, diff_atts)

    attr_map.append(attr_list)


def print_attrs(attrs, file):
    for attr in attrs:
        if attr.value_type == numbers.Integral:
            vtype = 'int'
        elif attr.__class__ == attributes.ChoiceAttribute:
            choices = ','.join([str(c) for c in attr.choices])
            vtype = f'list [{choices}]'
        else:
            vtype = attr.value_type.__name__ if hasattr(attr.value_type, '__name__') else str(attr.value_type)

        if attr.default is None:
            file.write('* ' + attr.name + ': ' + vtype + '\n\n')
        else:
            file.write('* ' + attr.name + ': ' + vtype + ' (Default: ' + str(attr.default) + ')\n\n')

        if attr.description is not None:
            file.write('  * ' + attr.description + '\n\n')


with open('attributes.rst', mode='w') as file:
    file.write('================\n')
    file.write('Layer attributes\n')
    file.write('================\n\n\n')

    for attr_list in attr_map:
        file.write(attr_list.cls_name + '\n')
        file.write('=' * len(attr_list.cls_name) + '\n')

        if len(attr_list.base_attrs) > 0:
            file.write('Base attributes\n')
            file.write('---------------\n')
            print_attrs(attr_list.type_attrs, file)

        if len(attr_list.type_attrs) > 0:
            file.write('Type attributes\n')
            file.write('---------------\n')
            print_attrs(attr_list.base_attrs, file)

        if len(attr_list.weight_attrs) > 0:
            file.write('Weight attributes\n')
            file.write('-----------------\n')
            print_attrs(attr_list.weight_attrs, file)

        if len(attr_list.config_attrs) > 0:
            file.write('Configurable attributes\n')
            file.write('-----------------------\n')
            print_attrs(attr_list.config_attrs, file)

        if len(attr_list.backend_attrs) > 0:
            file.write('Backend attributes\n')
            file.write('-----------------------\n')
            for backend, backend_attrs in attr_list.backend_attrs.items():
                file.write(backend + '\n')
                file.write('^' * len(backend) + '\n')
                print_attrs(backend_attrs, file)
