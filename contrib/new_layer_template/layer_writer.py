import argparse
import os

import yaml

# Parameters

# config = {'pyconfig' : {
# 	'layername': "test",
# 	'outputshape': 2,
# 	'attrlist' : {'Attribute': ['n_filt','epsilon','ops'],
#                   'WeightAttribute': ['gamma','beta'],
#                   'TypeAttribute': ['gamma','beta']},
# 	'genattrtypes' : ['static const unsigned','float','ap_fixed<32,10>'],
# 	'otherargs' : ['gamma','beta'],
# 	'attrdef' :
# """if len(input_shapes[0]) == 2:
#     layer['n_filt'] = -1
# elif len(input_shapes[0]) == 3:
#     layer['n_filt'] = input_shapes[0][2]
# elif len(input_shapes[0]) == 4:
#     layer['n_filt'] = input_shapes[0][3]
# """
# }}


def pythonWriter(filedir, config):
    f = open(os.path.join(filedir, 'py_impl_template.txt'))
    fout = open(f'{config["layername"]}.py', 'w')
    indent = '    '
    dindent = indent + indent
    for line in f.readlines():
        if 'Layernamelong' in line:
            newline = line.replace('Layernamelong', config["layerlongname"])
        elif 'Layername' in line:
            newline = line.replace('Layername', config["layername"].capitalize())
            if 'layername' in line:
                newline = newline.replace('layername', config["layername"])
        elif 'layername' in line:
            newline = line.replace('layername', config["layername"])

        # HLS4ML Layer implementation
        elif 'Attrs' in line:
            newline = ''
            newline += dindent + 'Attribute(\'n_in\'),\n'
            for attrtype in config["attrlist"].keys():
                if len(config["attrlist"][attrtype]) == 0:
                    break
                if attrtype == 'Attribute':
                    for i, att in enumerate(config["attrlist"][attrtype]):
                        newline += (
                            dindent
                            + f'{attrtype}(\'{att}\', value_type={config["genattrtypes"][i]},'
                            + f' default={config["genattrdef"][i]}),\n'
                        )
                else:
                    for att in config["attrlist"][attrtype]:
                        newline += dindent + f'{attrtype}(\'{att}\'),\n'
            newline = newline[:-2] + '\n'
        # Input output shape missing
        elif 'getweights' in line:
            newline = ''
            for att in config["attrlist"]['WeightAttribute']:
                newline += (
                    dindent
                    + f'{att} = self.model.get_weights_data(self.name, \'{att}\')\n'
                    + dindent
                    + f'self.add_weights_variable(name=\'{att}\', var_name=\'{att}{{index}}\', data={att}) \n'
                )

        # Struct config template
        elif 'structtemplate' in line:
            newline = ''
            for att in config["attrlist"]['WeightAttribute']:
                newline += indent + f'typedef {{{att}_t.name}} {att}_t;\n'
            assert len(config["attrlist"]['Attribute']) == len(config["genattrtypes"])
            for i, att in enumerate(config["attrlist"]['Attribute']):
                newline += indent + f'typedef {config["genattrtypes"][i]} {att}_t;\n'
            newline += indent + 'static const unsigned n_in = {n_in};'

        elif 'functemp' in line:
            newline = (
                f'{config["layername"]}_function_template  = (\n'
                + indent
                + f'\'nnet::{config["layername"]}<{{data_t}}, {{res_t}}, {{CONFIG_T}}>({{data}}, {{res}} '
            )
            if len(config["otherargs"]) != 0:
                for ar in config["otherargs"]:
                    newline += f',{{{ar}}}'
            newline += ' );\'\n)'
            newline = line.replace('functemp', newline)

        elif 'otherparms' in line:
            newline = ''
            for att in config["attrlist"]['WeightAttribute']:
                newline += dindent + f'params[\'{att}\'] = node.get_weights(\'{att}\').name\n'
            for att in config["attrlist"]['Attribute']:
                newline += dindent + f'params[\'{att}\'] = node.get_attr(\'{att}\')\n'

        elif 'outdim' in line:
            newline = ''
            if config["outputshape"] == 0:
                newline += 'layer[\'n_out\'] = layer[\'n_in\']'
            else:
                newline += f'layer[\'n_out\'] = {config["outputshape"]}'
            newline = line.replace('outdim', newline)

        elif 'attrdef' in line:
            newline = ''
            if config["attrdef"] != 0:
                attrdef = config["attrdef"].split('\n')
                for j in attrdef:
                    newline += indent + j + '\n'
            newline = line.replace('attrdef', newline)

        # Just copy line
        else:
            newline = line
        fout.write(newline)
    fout.close()
    return f'{config["layername"]}.py'


def hlsWriter(filedir, hlsconfig, pyconfig):
    f = open(os.path.join(filedir, 'hls_impl_template.h'))
    fout = open(f'{config["layername"]}.h', 'w')
    indent = '    '
    # dindent = indent + indent

    for line in f.readlines():
        if 'LAYERNAME' in line:
            newline = line.replace('LAYERNAME', pyconfig["layername"].upper())
        elif 'layername' in line:
            newline = line.replace('layername', pyconfig["layername"])

        elif 'deftypedef' in line:
            newline = ''
            for att in pyconfig["otherargs"]:
                newline += indent + f'typedef float {att}_t;\n'
            # assert len(pyconfig["attrlist"]['Attribute']) == len(pyconfig["genattrtypes"])
            # for i,att in enumerate(pyconfig["attrlist"]['Attribute']):
            # 	newline += indent + f'typedef {pyconfig["genattrtypes"][i]} {att}_t;\n'
            # newline += indent + f'static const unsigned n_in = {{n_in}};'
        elif 'exampledef' in line:
            newline = ''
            newline += indent + '//e.g. ' + f'{pyconfig["genattrtypes"][0]} {pyconfig["attrlist"]["Attribute"][0]} = 0;\n'
        elif 'arglist' in line:
            newline = ''
            for att in pyconfig["otherargs"]:
                newline += indent + f'typename CONFIG_T::{att}_t {att};\n'
        else:
            newline = line
        fout.write(newline)
    fout.close()
    return f'{config["layername"]}.h'


def initWriter(filedir, pyconfig):
    f = open(os.path.join(filedir, 'init_template.txt'))
    fout = open('__init__.py', 'w')
    # indent = '    '
    # dindent = indent + indent

    for line in f.readlines():
        if 'Layernamefolder' in line:
            newline = line.replace('Layernamefolder', pyconfig["layername"].capitalize() + "_layer")
            newline = newline.replace('Layername', config["layername"].capitalize())
            newline = newline.replace('layername', pyconfig["layername"])
        elif 'Layernamelong' in line:
            newline = line.replace('Layernamelong', config["layerlongname"])
            if 'layername' in line:
                newline = newline.replace('layername', pyconfig["layername"])
            if 'Layername' in line:
                newline = newline.replace('Layername', config["layername"].capitalize())
        elif 'Layername' in line:
            newline = line.replace('Layername', config["layername"].capitalize())
            if 'layername' in line:
                newline = newline.replace('layername', pyconfig["layername"])
        elif 'layername' in line:
            newline = line.replace('layername', pyconfig["layername"])
        else:
            newline = line
        fout.write(newline)
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, help='Relative path of config yaml file', default="config.yaml")
    args = parser.parse_args()

    filedir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
        hlsconfig = config["Cconfig"]
        config = config["Pyconfig"]
    pyfile = pythonWriter(filedir, config)
    hlsfile = hlsWriter(filedir, hlsconfig, config)
    initWriter(filedir, config)
    layerdir = config["layername"].capitalize() + "_layer"
    if not os.path.exists(layerdir):
        os.mkdir(layerdir)
    os.rename(pyfile, layerdir + '/' + pyfile)
    os.rename(hlsfile, layerdir + '/' + hlsfile)
    os.rename('__init__.py', layerdir + '/' + "__init__.py")
