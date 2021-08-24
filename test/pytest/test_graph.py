import hls4ml
import numpy as np
import pytest

class Reader:
    def get_weights_data(self, name, var):
        w = 2 if var =='kernel' else 1
        return np.array([w])
reader = Reader()

def base_model(output_dir='hls4mlprj_graph_base_model'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input', 'input_shape' : [1]},
            {'class_name' : 'Dense', 'name' : 'layer0', 'n_in' : 1, 'n_out' : 1},
            {'class_name' : 'Dense', 'name' : 'layer1', 'n_in' : 1, 'n_out' : 1}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = 'io_parallel'
  model = hls4ml.model.HLSModel(config, reader, layers)
  return model

def branched_model(output_dir='hls4mlprj_graph_branched_model'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input', 'input_shape' : [1]},
            {'class_name' : 'Dense', 'name' : 'layer0', 'n_in' : 1, 'n_out' : 1},
            {'class_name' : 'Dense', 'name' : 'layer1l', 'n_in' : 1, 'n_out' : 1, 'inputs' :['layer0']},
            {'class_name' : 'Dense', 'name' : 'layer1r', 'n_in' : 1, 'n_out' : 1, 'inputs' : ['layer0']},
            {'class_name' : 'Concatenate', 'name' : 'concat', 'axis' : 0, 'op' : 'concatenate1d', 'inputs' : ['layer1l', 'layer1r']}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = 'io_parallel'
  model = hls4ml.model.HLSModel(config, reader, layers)
  return model

def skip_model_l(output_dir='hls4mlprj_graph_skip_model_l'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input', 'input_shape' : [1]},
            {'class_name' : 'Dense', 'name' : 'layer0', 'n_in' : 1, 'n_out' : 1},
            {'class_name' : 'Dense', 'name' : 'layer1', 'n_in' : 1, 'n_out' : 1, 'inputs' :['layer0']},
            {'class_name' : 'Concatenate', 'name' : 'concat', 'axis' : 0, 'op' : 'concatenate1d', 'inputs' : ['layer1', 'layer0']}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = 'io_parallel'
  model = hls4ml.model.HLSModel(config, reader, layers)
  return model

def skip_model_r(output_dir='hls4mlprj_graph_skip_model_r'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input', 'input_shape' : [1]},
            {'class_name' : 'Dense', 'name' : 'layer0', 'n_in' : 1, 'n_out' : 1},
            {'class_name' : 'Dense', 'name' : 'layer1', 'n_in' : 1, 'n_out' : 1, 'inputs' :['layer0']},
            {'class_name' : 'Concatenate', 'name' : 'concat', 'axis' : 0, 'op' : 'concatenate1d', 'inputs' : ['layer0', 'layer1']}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = 'io_parallel'
  model = hls4ml.model.HLSModel(config, reader, layers)
  return model

@pytest.mark.parametrize('parameters', [(base_model, [3]),
                                        (branched_model, [3, 3]),
                                        (skip_model_l, [3, 1]),
                                        (skip_model_r, [1, 3])])
def test_base_models(parameters):
  model, expected = parameters[0](), np.array(parameters[1])
  model.compile()
  odir = model.config.config['OutputDir']
  hls4ml.utils.plot_model(model, show_shapes=True, show_precision=True, to_file='{}/model.png'.format(odir))
  X = np.zeros((1,1))
  y = model.predict(X)
  np.testing.assert_array_equal(y, expected)

def do_insert(model, node, layers):
  after, before = node[0], node[1]
  new_node = model.make_node('Dense', 'layer2', {'n_in' : 1, 'n_out' : 1}, [after])
  if not before is None:
    before = [x for x in model.graph.values() if x.name == before][0]
  model.insert_node(new_node, before=before)
  iInsert = np.argwhere(layers == after)[0][0] + 1
  layers = np.insert(layers, iInsert, 'layer2')
  return model, layers

def do_remove(model, node, layers):
  node_obj = [n for n in list(model.get_layers()) if n.name == node][0]
  model.remove_node(node_obj)
  iRemove = np.argwhere(layers == node)[0][0]
  layers = np.delete(layers, iRemove)
  return model, layers

def do_replace(model, node, layers):
  old_node = model.graph.get(node)
  new_node = model.make_node('Dense', 'layer2', {'n_in' : 1, 'n_out' : 1}, old_node.inputs)
  model.replace_node(old_node, new_node)
  iInsert = np.argwhere(layers == node)[0][0]
  layers = np.delete(layers, iInsert)
  layers = np.insert(layers, iInsert, 'layer2')
  return model, layers

graph_ops = {'insert'  : do_insert,
             'remove'  : do_remove,
             'replace' : do_replace}

@pytest.mark.parametrize('parameters', [(base_model, 'insert', ('layer0_input', None), [7], False),      # 0
                                        (base_model, 'insert', ('layer0', None), [7], False),            # 1
                                        (base_model, 'insert', ('layer1', None), [7], False),            # 2
                                        (base_model, 'remove', 'layer0', [1], False),                    # 3   
                                        (base_model, 'remove', 'layer1', [1], False),                    # 4
                                        (base_model, 'replace', 'layer0', [3], False),                   # 5
                                        (base_model, 'replace', 'layer1', [3], False),                   # 6
                                        (branched_model, 'insert', ('layer0', None), [7,7], False),      # 7
                                        (branched_model, 'insert', ('layer0', 'layer1l'), [7,3], False), # 8
                                        (branched_model, 'insert', ('layer0', 'layer1r'), [3,7], False), # 9
                                        (branched_model, 'insert', ('layer1l', None), [7,3], False),     # 10
                                        (branched_model, 'insert', ('layer1r', None), [3,7], False),     # 11
                                        (branched_model, 'remove', 'layer0', [1,1], False),              # 12
                                        (branched_model, 'remove', 'layer1l', [1,3], False),             # 13
                                        (branched_model, 'remove', 'layer1r', [3,1], False),             # 14
                                        (branched_model, 'replace', 'layer0', [3,3], False),             # 15
                                        (branched_model, 'replace', 'layer1l', [3,3], False),            # 16
                                        (branched_model, 'replace', 'layer1r', [3,3], True),             # 17
                                        (skip_model_l, 'insert', ('layer0', None), [7, 3], False),       # 18
                                        (skip_model_l, 'insert', ('layer0', 'layer1'), [7, 1], False),   # 19
                                        (skip_model_l, 'insert', ('layer0', 'concat'), [3, 3], False),   # 20
                                        (skip_model_l, 'insert', ('layer1', None), [7, 1], False),       # 21
                                        (skip_model_l, 'remove', 'layer0', [1, 0], False),               # 22
                                        (skip_model_l, 'remove', 'layer1', [1, 1], False),               # 23
                                        (skip_model_l, 'replace', 'layer0', [3, 1], False),              # 24
                                        (skip_model_l, 'replace', 'layer1', [3, 1], True),               # 25
                                        (skip_model_r, 'insert', ('layer0', None), [3, 7], False),       # 26
                                        (skip_model_r, 'insert', ('layer0', 'layer1'), [1, 7], False),   # 27
                                        (skip_model_r, 'insert', ('layer0', 'concat'), [3, 3], False),   # 28
                                        (skip_model_r, 'insert', ('layer1', None), [1, 7], False),       # 29
                                        (skip_model_r, 'remove', 'layer0', [0, 1], False),               # 30
                                        (skip_model_r, 'remove', 'layer1', [1, 1], False),               # 31
                                        (skip_model_r, 'replace', 'layer0', [1, 3], False),              # 32
                                        (skip_model_r, 'replace', 'layer1', [1, 3], True)])              # 33
def test_graph_manipulation(parameters):
  model, op, node, expected, skip_layers_check = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
  odir = 'hls4mlprj_graph_{}_{}_{}'.format(model.__name__, op, node)
  model = model(odir)
  original_layers = np.array([layer.name for layer in list(model.get_layers())])
  model, expected_layers = graph_ops[op](model, node, original_layers)
  model.compile()
  hls4ml.utils.plot_model(model, show_shapes=True, show_precision=True, to_file='{}/model.png'.format(odir))
  X = np.zeros((1,1))
  y = model.predict(X)
  # check the output
  expected = np.array(expected)
  np.testing.assert_array_equal(y, expected)
  # check the order
  actual_layers = np.array([layer.name for layer in list(model.get_layers())])
  if not skip_layers_check: # skip check for this model since order changes
    np.testing.assert_array_equal(expected_layers, actual_layers)
