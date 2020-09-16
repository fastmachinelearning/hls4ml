import os
from shutil import copyfile
import numpy as np
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.model.hls_model import IntegerPrecisionType, FixedPrecisionType

class PynqWriter(VivadoWriter):

    def next_axi_type(self, p):
        ''' Return a new type with the width rounded to the next factor of 8 up to p's width
            Args:
                p : IntegerPrecisionType or FixedPrecisionType
            Returns:
                An IntegerPrecisionType or FixedPrecisionType with the width rounder up to the next factor of 8
                of p's width. Other parameters (fractional bits, extra modes) stay the same.
        '''
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode, p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)


    def write_axi_wrapper(self, model):
        ''' Write a top level HLS C++ file to wrap the hls4ml project with AXI interfaces
            Args:
                model : The HLSModel to write the wrapper for
        '''

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        assert len(model_inputs) == 1, "Only models with one input tensor are currently supported by PynqBackend"
        assert len(model_outputs) == 1, "Only models with one output tensor are currently supported by PynqBackend"
        inp = model_inputs[0]
        out = model_outputs[0]
        inp_axi_t = self.next_axi_type(inp.type.precision)
        out_axi_t = self.next_axi_type(inp.type.precision)

        indent = '    '

        #######################
        ## myproject_axi.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/pynq/myproject_axi.h'),'r')
        fout = open('{}/firmware/{}_axi.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif '//hls-fpga-machine-learning insert include' in line:
                newline = '#include "{}.h"\n'.format(model.config.get_project_name())
            elif 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert definitions' in line:
                newline = ''
                newline += 'static const unsigned N_IN = {};\n'.format(inp.size())
                newline += 'static const unsigned N_OUT = {};\n'.format(out.size())
                newline += 'typedef {} input_axi_t;\n'.format(inp_axi_t)
                newline += 'typedef {} output_axi_t;\n'.format(out_axi_t)
                #newline += 'typedef {} input_t;\n'.format(inp.type.precision)
                #newline += 'typedef {} output_t ;\n'.format(out.type.precision)
                #newline += 'typedef {} input_axi_t;\n'.format(inp_axi_t)
                #newline += 'typedef {} output_axi_t;\n'.format(out_axi_t)
                #newline += 'typedef {} input_t;\n'.format(inp.type.precision)
                #newline += 'typedef {} output_t;\n'.format(out.type.precision)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

        #######################
        ## myproject_axi.cpp
        #######################

        f = open(os.path.join(filedir,'../templates/pynq/myproject_axi.cpp'),'r')
        fout = open('{}/firmware/{}_axi.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')
        
        io_type = model.config.get_config_value("IOType")

        for line in f.readlines():
            if 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert include' in line:
                newline = '#include "{}_axi.h"\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert local vars' in line:
                if io_type == 'io_parallel':
                    newline = ''
                    newline += indent + inp.type.name + ' in_local[N_IN];\n'
                    newline += indent + out.type.name + ' out_local[N_OUT];\n'
                elif io_type == 'io_stream':
                    newline = ''
                    newline += indent + 'hls::stream<' + inp.type.name + '> in_local("input_1");\n'
                    newline += indent + 'hls::stream<' + out.type.name + '> out_local("output_1");\n\n'
                    newline += indent + '#pragma HLS STREAM variable=in_local depth=N_IN\n'
                    newline += indent + '#pragma HLS STREAM variable=out_local depth=N_OUT\n'
            elif '//hls-fpga-machine-learning insert call' in line:
                newline = indent + '{}(in_local, out_local, in_size, out_size);\n'.format(model.config.get_project_name())         
            elif '//hls-fpga-machine-learning insert interface' in line:
                if model.config.interface == 's_axilite':
                    newline = ''
                    newline += indent + '#pragma HLS INTERFACE ap_ctrl_none port=return\n'
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=in\n'
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=out\n'
                elif model.config.interface == 'm_axi':
                    newline = ''
                    newline += indent + '#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS\n'
                    newline += indent + '#pragma HLS INTERFACE m_axi depth=in_size port=in offset=slave bundle=IN_BUS\n'
                    newline += indent + '#pragma HLS INTERFACE m_axi depth=out_size port=out offset=slave bundle=OUT_BUS\n'
            elif '//hls-fpga-machine-learning insert enqueue' in line:
                io_type = model.config.get_config_value("IOType")
                if io_type == 'io_parallel':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_IN; i++){\n'
                    newline += indent + indent + '#pragma HLS UNROLL\n'
                    newline += indent + indent + 'in_local[i] = in[i]; // Read input with cast\n'
                    newline += indent + '}\n'
                elif io_type == 'io_stream':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_IN / {input_t}::size; ++i) {{\n'
                    #newline += indent + indent + '#pragma HLS PIPELINE\n'
                    newline += indent + indent + '{input_t} ctype;\n'
                    newline += indent + indent + '#pragma HLS DATA_PACK variable=ctype\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {input_t}::size; j++) {{\n'
                    #newline += indent + indent + indent + '#pragma HLS UNROLL\n'
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
                    newline += indent + indent + '#pragma HLS UNROLL\n'
                    newline += indent + indent + 'out[i] = out_local[i]; // Write output with cast\n'
                    newline += indent + '}\n'
                elif io_type == 'io_stream':
                    newline = ''
                    newline += indent + 'for(unsigned i = 0; i < N_OUT / {result_t}::size; ++i) {{\n'
                    #newline += indent + indent + '#pragma HLS PIPELINE\n'
                    newline += indent + indent + '{result_t} ctype = out_local.read();\n'
                    newline += indent + indent + 'for(unsigned j = 0; j < {result_t}::size; j++) {{\n'
                    #newline += indent + indent + indent + '#pragma HLS UNROLL\n'
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
        Modify the build_prj.tcl script to add the extra wrapper files and set the top function
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        oldfile = '{}/build_prj.tcl'.format(model.config.get_output_dir())
        newfile = '{}/build_prj_axi.tcl'.format(model.config.get_output_dir())
        f = open(oldfile,'r')
        fout = open(newfile, 'w')

        for line in f.readlines():
            if 'set_top' in line:
                newline = line[:-1] + '_axi\n' # remove the newline from the line end and append _axi for the new top
                #newline += 'add_files firmware/{}_axi.h -cflags "-std=c++0x"\n'.format(model.config.get_project_name())
                newline += 'add_files firmware/{}_axi.cpp -cflags "-std=c++0x"\n'.format(model.config.get_project_name())
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)

    def write_board_script(self, model):
        '''
        Write the tcl scripts to create a Vivado IPI project for the Pynq
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.join(filedir,'../templates/pynq/pynq_design.tcl'), '{}/pynq_design.tcl'.format(model.config.get_output_dir()))
        f = open('{}/project.tcl'.format(model.config.get_output_dir()),'w')
        f.write('variable myproject\n')
        f.write('set myproject "{}"\n'.format(model.config.get_project_name()))
        
    def write_hls(self, model):
        '''
        Write the HLS project. Calls the VivadoBackend writer, and extra steps for Pynq/AXI interface
        '''
        super(PynqWriter, self).write_hls(model)
        self.write_axi_wrapper(model)
        self.modify_build_script(model)
        self.write_board_script(model)


