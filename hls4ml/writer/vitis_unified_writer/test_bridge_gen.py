import os
from meta import VitisUnifiedWriterMeta
import meta_gen as mg

def write_bridge_multigraph(meta, model):
    filedir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(filedir, '../templates/vitis_unified/myproject_bridge.cpp'))
    fout = open(f"{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp", 'w')
    model_inputs = model.graphs[0].get_input_variables()
    model_outputs = model.graphs[-1].get_output_variables()
    model_brams = [var for var in model.graphs[0].get_weight_variables() if var.storage.lower() == 'bram']

    indent = '    '

    for line in f.readlines():
        newline = ''
        if 'MYPROJECT' in line:
            newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
        elif 'firmware/myproject' in line:
            for graph_idx, g in enumerate(model.graphs):
                newline += '#undef DEFINES_H_\n'
                if len(g.outputs) == 1:
                    newline += '#define result_t ' + 'result_graph' + str(graph_idx + 1) + '_t\n'
                newline += line.replace('myproject', format(model.graphs[graph_idx].config.get_project_name()))
                if len(g.outputs) == 1:
                    newline += (
                        'typedef result_graph' + str(graph_idx + 1) + '_t graph' + str(graph_idx + 1) + '_result_t;\n'
                    )
                    newline += '#undef result_t\n\n' if graph_idx < len(model.graphs) - 1 else '\n'
            newline += '\n'
        elif 'myproject' in line:
            newline = line.replace('myproject', format(model.config.get_project_name()))

        elif '// hls-fpga-machine-learning insert bram' in line:
            newline = line
            for bram in model_brams:
                newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

        elif '// hls-fpga-machine-learning insert header' in line:
            dtype = line.split('#', 1)[1].strip()
            inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
            outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

            newline = ''
            newline += indent + inputs_str + ',\n'
            newline += indent + outputs_str + '\n'

        elif '// hls-fpga-machine-learning insert wrapper' in line:
            dtype = line.split('#', 1)[1].strip()
            newline = ''
            for inp in model_inputs:

                if meta.vitis_unified_config.isFreeInterimInput():
                    newline += indent + f"hls::stream<{inp.type.name}> {meta.getWrapperPortName(inp, True)};\n"
                    newline += indent + "nnet::convert_data_pkt<{srcType}, {underlying_data_T}, {data_T}, {sz}>({inpRaw}, {inp_wrapper});\n".format(
                        srcType           =dtype,
                        underlying_data_T=inp.type.name,
                        data_T=meta.get_axi_wrapper_type(inp),
                        sz=str(inp.size()),
                        inpRaw=inp.name,
                        inp_wrapper=meta.getWrapperPortName(inp, True),
                    )
                else:
                    newline += indent + f"hls::stream<{meta.getDmaTypeName()}> " + meta.getWrapperPortName(inp, True) + ";\n"
                    newline += indent + "nnet::convert_data<{dtype}, {dtype}, {sz}>({inpRaw}, {inp_wrapper});\n".format(
                        dtype=dtype,
                        sz=str(inp.size()),
                        inpRaw=inp.name,
                        inp_wrapper=meta.getWrapperPortName(inp, True),
                    )

                # newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                # newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                #     dtype, i.type.name, i.size_cpp(), i.name, i.name
                #)
            newline += '\n'


            for idx, g in enumerate(model.graphs):
                for out in g.get_output_variables():
                    outStreamName = meta.getWrapperPortName(out, False)
                    outStreamType = meta.get_axi_wrapper_type(out) if meta.vitis_unified_config.isFreeInterimOutput() else meta.getDmaTypeName()
                    newline += indent + f"hls::stream<{outStreamType}> {outStreamName}(\"{outStreamName}\");\n"
                    # definition = o.definition_cpp(name_suffix='_ap')
                    # if len(g.outputs) == 1:
                    #     parts = definition.split(' ', 1)
                    #     datatype = 'graph' + str(idx + 1) + '_result_t'
                    #     if parts[0].startswith('hls::stream'):
                    #         modified_definition = 'hls::stream<' + datatype + '> ' + parts[1]
                    #     else:
                    #         modified_definition = datatype + ' ' + parts[1]
                    #     newline += indent + f"{modified_definition};\n"
                    # else:
                    #     newline += indent + f"{definition};\n"

            newline += '\n'

            top_level = ''
            myOutputNextInput = []
            #output_vars = ''
            for idx, g in enumerate(model.graphs):
                # if idx == 0:
                #     input_vars = ','.join([i.name + '_ap' for i in g.get_input_variables()])
                # else:
                #     input_vars = output_vars
                # bram_vars = ','.join(
                #     [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                # )
                # output_vars = ','.join([o.name + '_ap' for o in g.get_output_variables()])
                # # Concatenate the input, output, and bram variables. Filter out empty/null values
                if idx == 0:
                    input_vars = [meta.getWrapperPortName(inp, True) for inp in g.get_input_variables()]
                else:
                    input_vars = myOutputNextInput.copy()

                output_vars = [meta.getWrapperPortName(out, False) for out in g.get_output_variables()]
                myOutputNextInput = output_vars.copy()
                bram_vars   = [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                allArgs = input_vars + output_vars + bram_vars
                all_vars = ', '.join(allArgs)
                top_level += indent + f"{g.config.get_project_name()}_axi({all_vars});\n"
            newline += top_level

            newline += '\n'

            for outIdx, o in enumerate(model_outputs):
                # if len(model.graphs[-1].outputs) == 1:
                #     newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                #         datatype, dtype, o.size_cpp(), o.name, o.name
                #     )
                # else:
                #     newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                #         o.type.name, dtype, o.size_cpp(), o.name, o.name
                #     )
                if meta.vitis_unified_config.isFreeInterimOutput():
                    newline += indent + (f"nnet::convert_data_pkt<{dtype}, {o.type.name}, "
                                         f"{meta.get_outputSizeArrName(model)}[{str(outIdx)}]>"
                                         f"({meta.getWrapperPortName(o, False)}, {o.name});\n")
                else:
                    newline += indent + (f"nnet::convert_data<{dtype}, {dtype}, {str(o.size())}>"
                                         f"({meta.getWrapperPortName(o, False)}, {o.name});\n")

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

        elif '// hls-fpga-machine-learning insert tb_input_writer' in line:
            funcs = [
                ("float", "dump_tb_inputs_float"),
                ("double", "dump_tb_inputs_double"),
            ]
            newline = ""
            for dtype, funcname in funcs:
                newline += f'void {funcname}(\n'
                newline += '    const char* output_path'
                for inp in model_inputs:
                    newline += f',\n    {dtype} {inp.name}[{inp.size_cpp()}]'
                newline += '\n) {\n\n'

                for inp in model_inputs:
                    decl = inp.definition_cpp(name_suffix='_ap').strip()
                    ap = inp.name + "_ap"
                    if decl.startswith("hls::stream"):
                        newline += f'    {decl};\n'
                    else:
                        newline += f'    {inp.type.name} {ap}[{inp.size_cpp()}];\n'
                    newline += (
                        f'    nnet::convert_data<{dtype}, {inp.type.name}, {inp.size_cpp()}>' f'({inp.name}, {ap});\n'
                    )
                newline += "\n"
                newline += f'    std::ofstream fout(std::string(output_path) + "/{inp.name}_input_data.txt");\n'

                for inp in model_inputs:
                    decl = inp.definition_cpp(name_suffix='_ap').strip()
                    dims = inp.shape

                    if decl.startswith("hls::stream"):
                        if len(dims) == 1:
                            N = dims[0]
                            newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                            newline += f'        auto temp = {inp.name}_ap.read();\n'
                            newline += (
                                f'        ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[0].range();\n'
                            )
                            newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                            newline += '    }\n'
                        else:
                            inputs_list = model.nn_config['inputs']
                            fifo_depth = next((e['fifo_depth'] for e in inputs_list if e['name'] == inp.name), None)
                            batch_size = next((e['batch_size'] for e in inputs_list if e['name'] == inp.name), None)
                            newline += f'    for(int r = 0; r < {fifo_depth}; r++) {{\n'
                            newline += f'        auto temp = {inp.name}_ap.read();\n'
                            newline += f'        for(int c = 0; c < {batch_size}; c++) {{\n'
                            newline += (
                                f'            ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[c].range();\n'
                            )
                            newline += (
                                f'            fout << bits.to_uint()' f' << (c+1<{batch_size} ? \' \' : \'\\n\');\n'
                            )
                            newline += '        }\n'
                            newline += '    }\n'
                    else:
                        ap = inp.name + "_ap"
                        N = inp.size_cpp()
                        newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                        newline += f'        ap_uint<{inp.type.name}::width> bits = ' f'{ap}[i].range();\n'
                        newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                        newline += '    }\n'
                newline += "    fout.close();\n"
                newline += "}\n"
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()