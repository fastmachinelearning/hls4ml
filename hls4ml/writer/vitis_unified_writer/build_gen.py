import os


def write_board_script(meta, model):
    '''
            Write the tcl scripts and kernel sources to create a Vivado IPI project for the VitisAcceleratorIPFlow
            '''
    ### I am not sure yet what it is
    # filedir = os.path.dirname(os.path.abspath(__file__))
    # copyfile(
    #     os.path.join(filedir, self.vitis_accelerator_ip_flow_config.get_tcl_file_path()),
    #     f'{model.config.get_output_dir()}/design.tcl',
    # )

    ###################
    # project.tcl
    ###################
    f = open(f'{model.config.get_output_dir()}/project.tcl', 'w')
    f.write('variable project_name\n')
    f.write(f'set project_name "{model.config.get_project_name()}"\n')
    f.write('variable backend\n')
    f.write('set backend "vitisacceleratoripflowpartial"\n')
    f.write('variable part\n')
    f.write("set part \"xc7z020clg400-1\"\n")
    #f.write(f'set part "{self.vitis_accelerator_ip_flow_config.get_part()}"\n')
    f.write('variable clock_period\n')
    f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
    f.write('variable clock_uncertainty\n')
    f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
    f.write('variable version\n')
    f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))
    # if self.vitis_accelerator_ip_flow_config.get_interface() == 'axi_stream':
    #     in_bit, out_bit = self.vitis_accelerator_ip_flow_config.get_io_bitwidth()
    #     f.write(f'set bit_width_hls_output {in_bit}\n')
    #     f.write(f'set bit_width_hls_input {out_bit}\n')
    f.close()
    return

def write_driver(meta, model):
    print("[partial reconfig] we are not supporting write_driver this yet")

def modify_build_script(meta, model):
    '''
    Modify the build_prj.tcl and build_lib.sh scripts to add the extra wrapper files and set the top function
    '''
    filedir = os.path.dirname(os.path.abspath(__file__))
    oldfile = f'{model.config.get_output_dir()}/build_prj.tcl'
    newfile = f'{model.config.get_output_dir()}/build_prj_axi.tcl'
    f = open(oldfile)
    fout = open(newfile, 'w')

    for line in f.readlines():
        if 'set_top' in line:
            newline = line[:-1] + '_axi\n'  # remove the newline from the line end and append _axi for the new top
            newline += f'add_files firmware/{model.config.get_project_name()}_axi.cpp -cflags "-std=c++0x"\n'
        elif f'{model.config.get_project_name()}_cosim' in line:
            newline = line.replace(
                f'{model.config.get_project_name()}_cosim',
                f'{model.config.get_project_name()}_axi_cosim',
            )
        elif '${project_name}.tcl' in line:
            newline = line.replace('${project_name}.tcl', '${project_name}_axi.tcl')
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()
    os.rename(newfile, oldfile)

    ###################
    # build_lib.sh
    ###################

    f = open(os.path.join(filedir, '../templates/vitis_unified/build_lib.sh'))
    fout = open(f'{model.config.get_output_dir()}/build_lib.sh', 'w')

    for line in f.readlines():
        line = line.replace('myproject', model.config.get_project_name())
        line = line.replace('mystamp', model.config.get_config_value('Stamp'))

        fout.write(line)
    f.close()
    fout.close()

def write_new_tar(meta, model):
    super().write_tar(model)