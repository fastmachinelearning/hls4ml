import os
from .meta import VitisUnifiedWriterMeta
from . import meta_gen as mg

def write_bridge(meta: VitisUnifiedWriterMeta, model):

    filedir = os.path.dirname(os.path.abspath(__file__))
    fin = open(os.path.join(filedir, '../templates/vitis_unified/myproject_bridge.cpp'))
    fout = open(f"{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp", 'w')

    model_inputs = model.get_input_variables()
    model_outputs = model.get_output_variables()
    model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

    indent = '    '

    for line in fin.readlines():
        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

        elif 'myproject' in line:
            newline = line.replace('myproject', format(model.config.get_project_name()))

        elif 'PROJECT_FILE_NAME' in line:
            newline = line.replace('PROJECT_FILE_NAME', format(mg.getGmemWrapperFileName(model)))

        elif '// hls-fpga-machine-learning insert bram' in line:
            newline = line
            for bram in model_brams:
                newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

        elif '// hls-fpga-machine-learning insert header' in line:
            dtype = line.split('#', 1)[1].strip()
            input_ios  = []
            output_ios = []

            for idx, inp in enumerate(model_inputs):
                input_ios.append(f"{dtype} {mg.getGmemIOPortName(inp, True, idx)}[{inp.size_cpp()}]")
                input_ios.append(f"int {mg.getGmemIOPortSizeName(inp, True, idx)}")
            for idx, out in enumerate(model_outputs):
                output_ios.append(f"{mg.getGmemIOPortName(out, False, idx)}[{out.size_cpp()}]")
                output_ios.append(f"int {mg.getGmemIOPortSizeName(out, False, idx)}")

            inputs_str = ', '.join(input_ios)
            outputs_str = ', '.join(output_ios)

            newline = ''
            newline += indent + inputs_str + ',\n'
            newline += indent + outputs_str + '\n'

        elif '// hls-fpga-machine-learning insert wrapper' in line:
            dtype = line.split('#', 1)[1].strip()
            newline = ''

            input_vars = []
            output_vars = []

            for idx, inp in enumerate(model_inputs):
                input_vars.append(mg.getGmemIOPortName(inp, True, idx))
                input_vars.append(mg.getGmemIOPortSizeName(inp, True, idx))
            for idx, out in enumerate(model_outputs):
                output_vars.append(mg.getGmemIOPortName(out, False, idx))
                output_vars.append(mg.getGmemIOPortSizeName(out, False, idx))

            inputs_str  = ', '.join(input_vars)
            outputs_str = ', '.join(output_vars)

            newline = ''
            newline += indent + inputs_str + ',\n'
            newline += indent + outputs_str + '\n'


        elif '// hls-fpga-machine-learning insert trace_outputs' in line:
            newline = ''
            for layer in model.get_layers():
                func = layer.get_attr('function_cpp', None)
                if func and model.config.trace_output and layer.get_attr('trace', False):
                    vars = layer.get_variables()
                    for var in vars:
                        newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                        )

        elif '// hls-fpga-machine-learning insert namespace' in line:
            newline = ''

            namespace = model.config.get_writer_config().get('Namespace', None)
            if namespace is not None:
                newline += indent + f'using namespace {namespace};\n'

        else:
            newline = line
        fout.write(newline)

    fin.close()
    fout.close()