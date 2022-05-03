import hls4ml
import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

test_root_path = Path(__file__).parent


class Reader:
    def get_weights_data(self, name, var):
        w = 2 if var =='kernel' else 1
        return np.array([w])
reader = Reader()

def base_model(output_dir='hls4mlprj_graph_base_model', iotype = 'io_parallel'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input', 'input_shape' : [1]},
            {'class_name' : 'Dense', 'name' : 'layer0', 'n_in' : 1, 'n_out' : 1},
            {'class_name' : 'Dense', 'name' : 'layer1', 'n_in' : 1, 'n_out' : 1}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = iotype
  config['Backend'] = 'Vivado'
  model = hls4ml.model.ModelGraph(config, reader, layers)
  return model

def branch_model(output_dir='hls4mlprj_graph_branch_model', iotype = 'io_parallel'):
  layers = [{'class_name' : 'Input', 'name' : 'layer0_input0', 'input_shape' : [1], 'inputs': 'input'},
            {'class_name' : 'Input', 'name' : 'layer0_input1', 'input_shape' : [1], 'inputs': 'input'},
            {'class_name' : 'Merge', 'name' : 'layer0', 'inputs' : ['layer0_input0', 'layer0_input1'], 'op' : 'add'},
            {'class_name' : 'Merge', 'name' : 'layer1', 'inputs' : ['layer0_input1', 'layer0'], 'op' : 'add'},
            {'class_name' : 'Merge', 'name' : 'layer2', 'inputs' : ['layer0_input0', 'layer1'], 'op' : 'add'}]
  config = {'HLSConfig':{'Model':{'Precision':'ap_fixed<32,16>','ReuseFactor' : 1}}}
  config['OutputDir'] = output_dir
  config['ProjectName'] = 'myprj'
  config['IOType'] = iotype
  model = hls4ml.model.ModelGraph(config, reader, layers, inputs=['layer0_input0', 'layer0_input1'])
  return model

def do_nop(model, node, layers):
  return model, layers

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
             'replace' : do_replace,
             'nop'     : do_nop}

@pytest.mark.parametrize('parameters', [(base_model, 'nop', None, [3], False),                           # 0
                                        (base_model, 'insert', ('layer0_input', None), [7], False),      # 1
                                        (base_model, 'insert', ('layer0', None), [7], False),            # 2
                                        (base_model, 'insert', ('layer1', None), [7], False),            # 3
                                        (base_model, 'remove', 'layer0', [1], False),                    # 4   
                                        (base_model, 'remove', 'layer1', [1], False),                    # 5
                                        (base_model, 'replace', 'layer0', [3], False),                   # 6
                                        (base_model, 'replace', 'layer1', [3], False)])                  # 7
@pytest.mark.parametrize('iotype', ['io_parallel', 'io_stream'])
def test_graph_manipulation(parameters, iotype):
  model, op, node, expected, skip_layers_check = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
  odir = str(test_root_path / 'hls4mlprj_graph_{}_{}_{}'.format(model.__name__, op, node))
  model = model(odir, iotype)
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

@pytest.mark.parametrize('iotype', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('batch', [1, 100])
def test_graph_branch(iotype, batch):
  odir = str(test_root_path / 'hls4mlprj_graph_branch_model_{}_batch{}'.format(iotype, batch))
  model = branch_model(odir, iotype)
  original_layers = np.array([layer.name for layer in list(model.get_layers())])
  model.compile()
  hls4ml.utils.plot_model(model, show_shapes=True, show_precision=True, to_file='{}/model.png'.format(odir))
  X0 = np.random.rand(batch, 1)
  X1 = np.random.rand(batch, 1)
  y_expected = 2*(X0+X1)
  y = model.predict([X0, X1]).reshape(y_expected.shape)
  # check the output
  np.testing.assert_allclose(y, y_expected, rtol=1, atol=2**-16)

@pytest.mark.parametrize('iotype', ['io_parallel', 'io_stream'])
def test_final_reshape(iotype):
  ''' Test case for a model with a Reshape as the final layer '''
  inputs = tf.keras.layers.Input(shape=(1,1,1)) # 1 input pixel
  conv = tf.keras.layers.Conv2D(6,1) # 6 filters, 1x1 kernel
  x = conv(inputs)
  conv.set_weights([np.linspace(1,6,6).reshape(1,1,1,6), np.zeros(6)]) # ascending int weights, 0 bias
  x = tf.keras.layers.Reshape((3,2))(x) # reshape the (1,1,6) output to (3,2)
  model = tf.keras.models.Model(inputs=inputs, outputs=x)

  # create the ModelGraph
  config = hls4ml.utils.config_from_keras_model(model, granularity='model')
  odir = str(test_root_path / 'hls4mlprj_graph_final_reshape_{}'.format(iotype))
  hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                         output_dir=odir,
                                                         backend='Vivado',
                                                         io_type = iotype,
                                                         hls_config=config)
  hls_model.compile()

  # Test on ascending integers. The weights mean that each output pixel/neuron has
  # a different value
  X = np.linspace(-4,4,9).reshape(9,1,1,1)
  y = model.predict(X)
  y_hls = hls_model.predict(X).reshape(y.shape)
  # because of integer inputs and integer weights, we can expect exact matching
  np.testing.assert_allclose(y, y_hls, rtol=0)




