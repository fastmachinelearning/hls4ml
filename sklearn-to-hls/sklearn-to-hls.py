from sklearn.externals import joblib
import numpy as np
import os
import argparse
import yaml
import sys
filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import bdt_writer, parse_config

def ensembleToDict(bdt):
  ensembleDict = {'max_depth' : bdt.max_depth, 'n_trees' : bdt.n_estimators,
                  'init_predict' : bdt.init_.predict(np.array([0]))[0][0],
                  'max_features' : bdt.max_features_, 'trees' : []}
  for tree in bdt.estimators_:
    tree = treeToDict(tree[0].tree_)
    tree = padTree(ensembleDict, tree)
    ensembleDict['trees'].append(tree)

  return ensembleDict

def treeToDict(tree):
  treeDict = {'feature' : tree.feature.tolist(), 'threshold' : tree.threshold.tolist(), 'value' : tree.value[:,0,0].tolist()}
  treeDict['children_left'] = tree.children_left.tolist()
  treeDict['children_right'] = tree.children_right.tolist()
  # add the parent index
  n = len(tree.children_left) # number of nodes
  parents = [0] * n
  for i in range(n):
    j = tree.children_left[i]
    if j != -1:
      parents[j] = i
    k = tree.children_right[i]
    if k != -1:
      parents[k] = i
  parents[0] = -1
  treeDict['parent'] = parents
  # Add the depth info
  treeDict['depth'] = [0] * n
  for i in range(n):
    depth = 0
    parent = treeDict['parent'][i]
    while parent != -1:
      depth += 1
      parent = treeDict['parent'][parent]
    treeDict['depth'][i] = depth
  return treeDict

def padTree(ensembleDict, treeDict):
  '''Pad a tree with dummy nodes if not perfectly balanced'''
  n_nodes = len(treeDict['children_left'])
  # while th tree is unbalanced
  while n_nodes != 2 ** (ensembleDict['max_depth'] + 1) - 1:
    for i in range(n_nodes):
      if treeDict['children_left'][i] == -1 and treeDict['depth'][i] != ensembleDict['max_depth']:
        treeDict['children_left'].extend([-1, -1])
        treeDict['children_right'].extend([-1, -1])
        treeDict['parent'].extend([i, i])
        treeDict['feature'].extend([-2, -2])
        treeDict['threshold'].extend([-2.0, -2.0])
        newDepth = treeDict['depth'][i] + 1
        treeDict['depth'].extend([newDepth, newDepth])
        iRChild = len(treeDict['children_left']) - 1
        iLChild = iRChild - 1
        treeDict['children_left'][i] = iLChild
        treeDict['children_right'][i] = iRChild
    n_nodes = len(treeDict['children_left'])
  return treeDict

def main():
  
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='')
  parser.add_argument("-c", action='store', dest='config',
                      help="Configuration file.")
  args = parser.parse_args()
  if not args.config: parser.error('A configuration file needs to be specified.')
  configDir  = os.path.abspath(os.path.dirname(args.config))
  yamlConfig = parse_config(args.config)
  if not os.path.isabs(yamlConfig['OutputDir']):
    yamlConfig['OutputDir'] = os.path.join(configDir, yamlConfig['OutputDir'])
  if not os.path.isabs(yamlConfig['sklearnPkl']):
    yamlConfig['sklearnPkl'] = os.path.join(configDir, yamlConfig['sklearnPkl'])

  if not (yamlConfig["IOType"] == "io_parallel"):
    raise Exception('ERROR: Invalid IO type (serial not yet supported)')

  ######################
  ##  Do translation
  ######################
  if not os.path.isdir("{}/firmware".format(yamlConfig['OutputDir'])):
    os.makedirs("{}/firmware".format(yamlConfig['OutputDir']))

  bdt = joblib.load(yamlConfig['sklearnPkl'])
  ensembleDict = ensembleToDict(bdt)
  bdt_writer(ensembleDict, yamlConfig)

if __name__ == "__main__":
  main()
