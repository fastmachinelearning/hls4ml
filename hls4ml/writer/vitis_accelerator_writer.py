
import glob
import os
from shutil import copy

from hls4ml.writer.vitis_writer import VitisWriter


class VitisAcceleratorWriter(VitisWriter):
    def __init__(self):
        super().__init__()

    def write_parameters_overrides(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include
                newline += '#include "defines.h"'

            elif '// hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'

            elif "// hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_build_script_backend_override(self, model):
        # project.tcl
        f = open(f'{model.config.get_output_dir()}/project.tcl', 'w')
        f.write('variable project_name\n')
        f.write(f'set project_name "{model.config.get_project_name()}"\n')
        f.write('variable backend\n')
        f.write('set backend "vitisaccelerator"\n')
        f.write('variable part\n')
        f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
        f.write('variable clock_period\n')
        f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
        f.write('variable clock_uncertainty\n')
        f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
        f.close()

    def write_kernel(self, model):
        """Write the Python-C++ kernel (myproject_kernel.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_accelerator/myproject_kernel.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_kernel.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([f'input_t *{i.name}' for i in model_inputs])
                outputs_str = ', '.join([f'result_t *{o.name}' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += '    uint32_t size\n'
            elif '// hls-fpga-machine-learning insert project top' in line:
                top_function_str = format(model.config.get_project_name())
                input_str = str(model_inputs[-1].name)
                output_str = str(model_outputs[-1].name)
                newline = indent + top_function_str + '(' + input_str + '_stream, ' + output_str + '_stream);\n'
            elif 'project_input' in line:
                # input = [i.name for i in model_inputs]
                newline = line.replace('project_input', str(model_inputs[-1].name))
            elif 'project_output' in line:
                # output = [o.name for o in model_outputs]
                newline = line.replace('project_output', str(model_outputs[-1].name))
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_host(self, model):
        """Write the Python-C++ kernel (myproject_host.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        from hls4ml.backends import VitisAcceleratorConfig
        vitis_accelerator_config = VitisAcceleratorConfig(model.config)

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_accelerator/myproject_host.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_host.cpp', 'w')
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif 'myproject_kernel' in line:
                newline = line.replace('myproject_kernel', format(model.config.get_project_name(), '_kernel'))
            elif 'output_dir' in line:
                newline = line.replace('output_dir', format(model.config.get_output_dir()))
            elif 'myplatform' in line:
                newline = line.replace('myplatform', format(vitis_accelerator_config.get_platform()))
            elif 'mylayer_out' in line:
                newline = line.replace('mylayer_out', format(model_outputs[-1].size_cpp())) 
            elif 'myinput' in line:
                newline = line.replace('myinput', format(model_inputs[-1].size_cpp()))
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_makefile(self, model):
        """Write the Python-C++ Makefile (Makefile)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        from hls4ml.backends import VitisAcceleratorConfig

        #        vivado_accelerator_config = VitisAcceleratorConfig(
        #            model.config, model.get_input_variables(), model.get_output_variables()
        #        )
        vitis_accelerator_config = VitisAcceleratorConfig(model.config)

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_accelerator/Makefile'))
        fout = open(f'{model.config.get_output_dir()}/Makefile', 'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif 'myproject_host' in line:
                newline = line.replace('myproject_host', format(model.config.get_project_name(), '_host'))
            elif 'myproject_kernel' in line:
                newline = line.replace('myproject_kernel', format(model.config.get_project_name(), '_kernel'))
            elif 'myplatform' in line:
                newline = line.replace('myplatform', format(vitis_accelerator_config.get_platform()))
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_accelerator_card_cfg(self, model):
        """Write the Python acceleratro card configuration (accelerator_card.cfg)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vitis_accelerator/accelerator_card.cfg'))
        fout = open(f'{model.config.get_output_dir()}/accelerator_card.cfg', 'w')

        from hls4ml.backends import VitisAcceleratorConfig

        #        vitis_accelerator_config = VitisAcceleratorConfig(
        #            model.config, model.get_input_variables(), model.get_output_variables()
        #        )
        vitis_accelerator_config = VitisAcceleratorConfig(model.config)

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif 'myproject_kernel' in line:
                newline = line.replace('myproject_kernel', format(model.config.get_project_name(), '_kernel'))
            elif 'myplatform' in line:
                newline = line.replace('myplatform', format(vitis_accelerator_config.get_platform()))
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils_overrides(self, model):
        """Override nnet_types.h pointer comparison

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/vitis_accelerator/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'
        copy(srcpath + "nnet_types.h", dstpath + "nnet_types.h")

    def write_hls(self, model):
        """
        Write the HLS project. Calls the steps from VivadoWriter, adapted for Vitis
        """
        super().write_hls(model)
        self.write_nnet_utils_overrides(model)
        self.write_build_script_backend_override(model)
        self.write_parameters_overrides(model)
        self.write_kernel(model)
        self.write_host(model)
        self.write_makefile(model)
        self.write_accelerator_card_cfg(model)
