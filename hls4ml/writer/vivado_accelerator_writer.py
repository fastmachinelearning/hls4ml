import os
from shutil import copyfile, copytree
from distutils.dir_util import copy_tree
from hls4ml.writer.vivado_writer import VivadoWriter

class VivadoAcceleratorWriter(VivadoWriter):

    def __init__(self):
        super().__init__()
        self.vivado_accelerator_config = None

    def write_axi_wrapper(self, model):
        ''' Write a top level HLS C++ file to wrap the hls4ml project with AXI interfaces
            Args:
                model : The ModelGraph to write the wrapper for
        '''
        inp_axi_t, out_axi_t, inp, out = self.vivado_accelerator_config.get_corrected_types()
        indent = '    '

        #######################
        ## myproject_axi.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado_accelerator/myproject_axi.h'), 'r')
        fout = open('{}/firmware/{}_axi.h'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif '//hls-fpga-machine-learning insert include' in line:
                newline = '#include "{}.h"\n'.format(model.config.get_project_name())
            elif 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert definitions' in line:
                newline = ''
                newline += 'static const unsigned N_IN = {};\n'.format(inp.size())
                newline += 'static const unsigned N_OUT = {};\n'.format(out.size())
                if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                    newline += 'typedef {} T_in;\n'.format(inp_axi_t)
                    newline += 'typedef {} T_out;\n'.format(out_axi_t)
                    newline += 'typedef struct in_struct {\n' + \
                               indent + 'T_in data;\n' + \
                               indent + 'ap_uint<1> last;\n' + \
                               indent + 'in_struct(const T_in& data, const ap_uint<1>& last){this->data = data; this->last = last;};\n' + \
                               indent + 'in_struct(){this->data = 0; this->last = 0;};\n' + \
                               indent + 'friend std::ostream& operator<<(std::ostream& stream, const in_struct& in)\n' + \
                               indent + '{ return stream << "{ data: " << in.data << ", last: " << in.last << " }" << std::endl; }\n' + \
                               indent + 'operator float() const {return this->data;}\n' + \
                               indent + 'operator double() const {return this->data;}\n' + \
                               indent + 'in_struct(float data) {this->data = data; this->last = 0;}\n' + \
                               indent + 'in_struct(double data) {this->data = data; this->last = 0;}\n' + \
                               '} input_axi_t;\n'
                    newline += 'typedef struct out_struct {\n' + \
                               indent + 'T_out data;\n' + \
                               indent + 'ap_uint<1> last;\n' + \
                               indent + 'out_struct(const T_out& data, const ap_uint<1>& last){this->data = data; this->last = last;};\n' + \
                               indent + 'out_struct(){this->data = 0; this->last = 0;};\n' + \
                               indent + 'friend std::ostream& operator<<(std::ostream& stream, const out_struct& out)\n' + \
                               indent + '{ return stream << "{ data: " << out.data << ", last: " << out.last << " }" << std::endl; }\n' + \
                               indent + 'operator float() const {return this->data;}\n' + \
                               indent + 'operator double() const {return this->data;}\n' + \
                               indent + 'out_struct(float data) {this->data = data; this->last = 0;}\n' + \
                               indent + 'out_struct(double data) {this->data = data; this->last = 0;}\n' + \
                               '} output_axi_t;\n'
                else:
                    newline += 'typedef {} input_axi_t;\n'.format(inp_axi_t)
                    newline += 'typedef {} output_axi_t;\n'.format(out_axi_t)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

        #######################
        ## myproject_axi.cpp
        #######################

        f = open(os.path.join(filedir, '../templates/vivado_accelerator/myproject_axi.cpp'), 'r')
        fout = open('{}/firmware/{}_axi.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),
                    'w')

        io_type = model.config.get_config_value("IOType")

        for line in f.readlines():
            if 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert include' in line:
                newline = '#include "{}_axi.h"\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert local vars' in line:
                newline = ''
                if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                    newline += indent + 'bool is_last = false;\n'
                if io_type == 'io_parallel':
                    newline += indent + inp.type.name + ' in_local[N_IN];\n'
                    newline += indent + out.type.name + ' out_local[N_OUT];\n'
                elif io_type == 'io_stream':
                    newline += indent + 'hls::stream<' + inp.type.name + '> in_local("input_1");\n'
                    newline += indent + 'hls::stream<' + out.type.name + '> out_local("output_1");\n\n'
                    newline += indent + '#pragma HLS STREAM variable=in_local depth={}\n'\
                        .format(model.get_input_variables()[0].pragma[1])
                    newline += indent + '#pragma HLS STREAM variable=out_local depth={}\n'\
                        .format(model.get_output_variables()[0].pragma[1])
            elif '//hls-fpga-machine-learning insert call' in line:
                newline = indent + '{}(in_local, out_local);\n'.format(
                    model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert interface' in line:
                if self.vivado_accelerator_config.get_interface() == 'axi_lite':
                    newline = ''
                    newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=in\n'
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=out\n'
                elif self.vivado_accelerator_config.get_interface() == 'axi_master':
                    newline = ''
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS\n'
                    newline += indent + '#pragma HLS INTERFACE m_axi depth={} port=in offset=slave bundle=IN_BUS\n'\
                        .format(model.get_input_variables()[0].pragma[1])
                    newline += indent + '#pragma HLS INTERFACE m_axi depth={} port=out offset=slave bundle=OUT_BUS\n'\
                        .format(model.get_output_variables()[0].pragma[1])
                elif self.vivado_accelerator_config.get_interface() == 'axi_stream':
                    newline = ''
                    newline += indent + '#pragma HLS INTERFACE axis port=in\n'
                    newline += indent + '#pragma HLS INTERFACE axis port=out\n'
                    newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                    if model.config.get_config_value("IOType") == 'io_stream':
                        newline += indent + '#pragma HLS DATAFLOW\n'
            elif '//hls-fpga-machine-learning insert enqueue' in line:
                io_type = model.config.get_config_value("IOType")
                if io_type == 'io_parallel':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_IN; i++){\n'
                    if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                        newline += indent + indent + '#pragma HLS PIPELINE\n'
                        newline += indent + indent + 'in_local[i] = in[i].data; // Read input with cast\n'
                        newline += indent + indent + 'is_last |= (in[i].last == 1)? true: false;\n'
                    else:
                        newline += indent + indent + '#pragma HLS UNROLL\n'
                        newline += indent + indent + 'in_local[i] = in[i]; // Read input with cast\n'
                    newline += indent + '}\n'
                elif io_type == 'io_stream':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_IN / {input_t}::size; ++i) {{\n'
                    # newline += indent + indent + '#pragma HLS PIPELINE\n'
                    newline += indent + indent + '{input_t} ctype;\n'
                    newline += indent + indent + '#pragma HLS DATA_PACK variable=ctype\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {input_t}::size; j++) {{\n'
                    # newline += indent + indent + indent + '#pragma HLS UNROLL\n'
                    if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                        newline += indent + indent + indent + 'ctype[j] = typename {input_t}::value_type(in[i * {input_t}::size + j].data);\n'
                        newline += indent + indent + indent + 'is_last |= (in[i * input_t::size + j].last == 1)? true : false;\n'
                    else:
                        newline += indent + indent + indent + 'ctype[j] = typename {input_t}::value_type(in[i * {input_t}::size + j]);\n'
                    newline += indent + indent + '}}\n'
                    newline += indent + indent + 'in_local.write(ctype);\n'
                    newline += indent + '}}\n'
                    newline = newline.format(input_t=inp.type.name)
            elif '//hls-fpga-machine-learning insert dequeue' in line:
                io_type = model.config.get_config_value("IOType")
                if io_type == 'io_parallel':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_OUT; i++){\n'
                    if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                        newline += indent + indent + '#pragma HLS PIPELINE\n'
                        newline += indent + indent + 'out[i].data = out_local[i]; // Write output with cast\n'
                        newline += indent + indent + 'out[i].last = (is_last && (i == N_OUT - 1))? true : false;\n'
                    else:
                        newline += indent + indent + '#pragma HLS UNROLL\n'
                        newline += indent + indent + 'out[i] = out_local[i]; // Write output with cast\n'
                    newline += indent + '}\n'
                elif io_type == 'io_stream':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_OUT / {result_t}::size; ++i) {{\n'
                    # newline += indent + indent + '#pragma HLS PIPELINE\n'
                    newline += indent + indent + '{result_t} ctype = out_local.read();\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {result_t}::size; j++) {{\n'
                    # newline += indent + indent + indent + '#pragma HLS UNROLL\n'
                    if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                        newline += indent + indent + indent + 'bool last = (is_last && (i * {result_t}::size + j == N_OUT - 1)) ? true : false;\n'
                        newline += indent + indent + indent + 'out[i * {result_t}::size + j] = output_axi_t(ctype[j], last);\n'
                    else:
                        newline += indent + indent + indent + 'out[i * {result_t}::size + j] = output_axi_t(ctype[j]);\n'
                    newline += indent + indent + '}}\n'
                    newline += indent + '}}\n'
                    newline = newline.format(result_t=out.type.name)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def modify_build_script(self, model):
        '''
        Modify the build_prj.tcl and build_lib.sh scripts to add the extra wrapper files and set the top function
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        oldfile = '{}/build_prj.tcl'.format(model.config.get_output_dir())
        newfile = '{}/build_prj_axi.tcl'.format(model.config.get_output_dir())
        f = open(oldfile, 'r')
        fout = open(newfile, 'w')

        for line in f.readlines():
            if 'set_top' in line:
                newline = line[:-1] + '_axi\n'  # remove the newline from the line end and append _axi for the new top
                newline += 'add_files firmware/{}_axi.cpp -cflags "-std=c++0x"\n'.format(
                    model.config.get_project_name())
            elif '{}_cosim'.format(model.config.get_project_name()) in line:
                newline = line.replace('{}_cosim'.format(model.config.get_project_name()), '{}_axi_cosim'.format(model.config.get_project_name()))
            elif '${myproject}.tcl' in line:
                newline = line.replace('${myproject}.tcl', '${myproject}_axi.tcl')
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir, '../templates/vivado_accelerator/build_lib.sh'), 'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())
            line = line.replace('mystamp', model.config.get_config_value('Stamp'))

            fout.write(line)
        f.close()
        fout.close()

    def write_wrapper_test(self, model):

        ###################
        # write myproject_test_wrapper.cpp
        ###################
        oldfile = '{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name())
        newfile = '{}/{}_test_wrapper.cpp'.format(model.config.get_output_dir(), model.config.get_project_name())

        f = open(oldfile, 'r')
        fout = open(newfile, 'w')

        inp = model.get_input_variables()[0]
        out = model.get_output_variables()[0]

        for line in f.readlines():
            if '{}.h'.format(model.config.get_project_name()) in line:
                newline = line.replace('{}.h'.format(model.config.get_project_name()),
                                       '{}_axi.h'.format(model.config.get_project_name()))
            elif inp.definition_cpp() in line:
                newline = line.replace(inp.definition_cpp(), 'input_axi_t inputs[N_IN]') #TODO instead of replacing strings, how about we use proper variables and their definition?
            elif out.definition_cpp() in line:
                newline = line.replace(out.definition_cpp(), 'output_axi_t outputs[N_OUT]')
            elif 'unsigned short' in line:
                newline = ''
            elif '{}('.format(model.config.get_project_name()) in line:
                indent_amount = line.split(model.config.get_project_name())[0]
                newline = indent_amount + '{}_axi(inputs,outputs);\n'.format(model.config.get_project_name())
            elif inp.size_cpp() in line or inp.name in line or inp.type.name in line:
                newline = line.replace(inp.size_cpp(), 'N_IN').replace(inp.name, 'inputs').replace(inp.type.name,
                                                                                                      'input_axi_t')
            elif out.size_cpp() in line or out.name in line or out.type.name in line:
                newline = line.replace(out.size_cpp(), 'N_OUT').replace(out.name, 'outputs').replace(out.type.name,
                                                                                                        'output_axi_t')
            else:
                newline = line
            if self.vivado_accelerator_config.get_interface() == 'axi_stream':
                if 'nnet::fill_zero' in line:
                    indent = line.split('n')[0]
                    newline = indent + 'inputs[N_IN-1].last = 1;\n'
                if 'copy_data' in line:
                    newline = newline.replace('copy_data', 'copy_data_axi')
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

        ###################
        # write myproject_bridge_wrapper.cpp
        ###################
        oldfile = '{}/{}_bridge.cpp'.format(model.config.get_output_dir(), model.config.get_project_name())
        newfile = '{}/{}_bridge_wrapper.cpp'.format(model.config.get_output_dir(), model.config.get_project_name())

        f = open(oldfile, 'r')
        fout = open(newfile, 'w')

        inp = model.get_input_variables()[0]
        out = model.get_output_variables()[0]

        for line in f.readlines():
            if '{}.h'.format(model.config.get_project_name()) in line:
                newline = line.replace('{}.h'.format(model.config.get_project_name()),
                                       '{}_axi.h'.format(model.config.get_project_name()))
            elif inp.definition_cpp(name_suffix='_ap') in line:
                newline = line.replace(inp.definition_cpp(name_suffix='_ap'),
                                       'input_axi_t {}_ap[N_IN]'.format(inp.name))
            elif out.definition_cpp(name_suffix='_ap') in line:
                newline = line.replace(out.definition_cpp(name_suffix='_ap'),
                                       'output_axi_t {}_ap[N_OUT]'.format(out.name))
            elif '{}('.format(model.config.get_project_name()) in line:
                indent_amount = line.split(model.config.get_project_name())[0]
                newline = indent_amount + '{}_axi({}_ap,{}_ap);\n'.format(model.config.get_project_name(), inp.name,
                                                                          out.name)
            elif inp.size_cpp() in line or inp.name in line or inp.type.name in line:
                newline = line.replace(inp.size_cpp(), 'N_IN').replace(inp.type.name, 'input_axi_t')
            elif out.size_cpp() in line or out.name in line or out.type.name in line:
                newline = line.replace(out.size_cpp(), 'N_OUT').replace(out.type.name, 'output_axi_t')
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

    def write_board_script(self, model):
        '''
        Write the tcl scripts and kernel sources to create a Vivado IPI project for the VivadoAccelerator
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.join(filedir, self.vivado_accelerator_config.get_tcl_file_path()),
                 '{}/design.tcl'.format(model.config.get_output_dir()))
        # Generic alveo board
        if self.vivado_accelerator_config.get_board().startswith('alveo'):
            src_dir=os.path.join(filedir, self.vivado_accelerator_config.get_krnl_rtl_src_dir())
            dst_dir= os.path.abspath(model.config.get_output_dir())+'/src'
            copy_tree(src_dir,dst_dir)
        f = open('{}/project.tcl'.format(model.config.get_output_dir()), 'w')
        f.write('variable myproject\n')
        f.write('set myproject "{}"\n'.format(model.config.get_project_name()))
        f.write('variable backend\nset backend vivadoaccelerator\n')
        if self.vivado_accelerator_config.get_board().startswith('alveo'):
             f.write('variable part\n')
             f.write('set part "{}"\n'.format(self.vivado_accelerator_config.get_part()))
        if self.vivado_accelerator_config.get_interface() == 'axi_stream':
            in_bit, out_bit = self.vivado_accelerator_config.get_io_bitwidth()
            f.write('set bit_width_hls_output {}\n'.format(in_bit))
            f.write('set bit_width_hls_input {}\n'.format(out_bit))
        f.close()

    def write_driver(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.join(filedir, self.vivado_accelerator_config.get_driver_path()),
                 ('{}/' + self.vivado_accelerator_config.get_driver_file()).format(model.config.get_output_dir()))
        
    def write_new_tar(self, model):
        os.remove(model.config.get_output_dir() + '.tar.gz')
        super(VivadoAcceleratorWriter, self).write_tar(model)

        
    def write_hls(self, model):
        """
        Write the HLS project. Calls the VivadoBackend writer, and extra steps for VivadoAccelerator/AXI interface
        """
        #TODO temporarily move config import here to avoid cyclic dependency, until config is moved to its own package
        from hls4ml.backends import VivadoAcceleratorConfig
        self.vivado_accelerator_config = VivadoAcceleratorConfig(model.config, model.get_input_variables(),
                                                                 model.get_output_variables())
        super(VivadoAcceleratorWriter, self).write_hls(model)
        self.write_board_script(model)
        self.write_driver(model)
        self.write_wrapper_test(model)
        self.write_axi_wrapper(model)
        self.modify_build_script(model)
        self.write_new_tar(model)

