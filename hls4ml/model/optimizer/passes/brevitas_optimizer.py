# Inserts Quant nodes into the model as needed for input/output quantization of layers in brevitas
import numpy as np

from hls4ml.model.optimizer import OptimizerPass


class BrevitasInputOutputOptimizer(OptimizerPass):
    '''Takes nodes parsed from brevitas and inserts Quant nodes into the model if necessary'''

    def match(self, node):
        needs_conversion = False
        if 'convert_io_from_brevitas' in node.attributes.keys():
            needs_conversion = node.attributes['convert_io_from_brevitas'] and (
                'output_quantization' in node.attributes.keys() or 'input_quantization' in node.attributes.keys()
            )

        return needs_conversion

    def transform(self, model, node):

        # See if Quant layer needs to be added for the output
        if 'output_quantization' in node.attributes.keys() and not len(node.attributes['output_quantization']) == 0:

            attributes = {}

            input = node.name
            # Other attributes
            attributes['narrow'] = node.attributes['output_quantization']['narrow']
            attributes['rounding_mode'] = node.attributes['output_quantization']['rounding_mode']
            attributes['signed'] = node.attributes['output_quantization']['signed']
            attributes['bitwidth'] = node.attributes['output_quantization']['bit_width']
            attributes['zeropt'] = node.attributes['output_quantization']['zeropoint']
            attributes['scale'] = np.array([node.attributes['output_quantization']['scale']])

            quant_node = model.make_node('Quant', f'quant_output_for_{node.get_attr("name")}', attributes, [input])
            quant_node.set_attr('name', f'quant_output_for_{node.get_attr("name")}')

            model.insert_node(quant_node)

            node.attributes['output_quantization'] = {}

        elif 'input_quantization' in node.attributes.keys() and not len(node.attributes['input_quantization']) == 0:

            attributes = {}

            input = node.inputs[0]
            # Other attributes
            attributes['narrow'] = node.attributes['input_quantization']['narrow']
            attributes['rounding_mode'] = node.attributes['input_quantization']['rounding_mode']
            attributes['signed'] = node.attributes['input_quantization']['signed']
            attributes['bitwidth'] = node.attributes['input_quantization']['bit_width']
            attributes['zeropt'] = node.attributes['input_quantization']['zeropoint']
            attributes['scale'] = np.array([node.attributes['input_quantization']['scale']])

            quant_node = model.make_node('Quant', f'quant_input_for_{node.get_attr("name")}', attributes, [input])
            quant_node.set_attr('name', f'quant_input_for_{node.get_attr("name")}')

            model.insert_node(quant_node)

            node.attributes['input_quantization'] = {}

        return True
