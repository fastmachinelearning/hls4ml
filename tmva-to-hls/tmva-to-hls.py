import xml.etree.ElementTree as ET
import numpy as np
import os
import argparse
import yaml
import sys
filedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(filedir, "..", "hls-writer"))
from hls_writer import bdt_writer, parse_config

def getOptionValue(bdt, optionName):
    for option in bdt.getroot().find('Options') :
        if optionName == option.get('name'):
            return option.text

def ensembleToDict(bdt):
  max_depth = int(getOptionValue(bdt, 'MaxDepth'))
  n_trees = int(getOptionValue(bdt, 'NTrees'))
  n_features = int(bdt.find('Variables').attrib['NVar'])
  n_classes = int(bdt.find('Classes').attrib['NClass'])
  ensembleDict = {'max_depth' : max_depth, 'n_trees' : n_trees,
                  'n_features' : n_features,
                  'n_classes' : n_classes, 'trees' : [],
                  'init_predict' : [0.],
                  'norm' : 0}
  for trees in bdt.find('Weights'):
    treesl = []
    #for tree in trees:
    # TODO find out how TMVA implements multi-class
    tree = trees
    weight = float(tree.attrib['boostWeight'])
    tree = treeToDict(bdt, tree)
    tree = padTree(ensembleDict, tree)
    treesl.append(tree)
    ensembleDict['trees'].append(treesl)
    ensembleDict['norm'] += weight
  # Invert the normalisation so FPGA can do '*' instead of '/'
  ensembleDict['norm'] = 1. / ensembleDict['norm'] 
  return ensembleDict

def addParentAndDepth(treeDict):
  n = len(treeDict['children_left']) # number of nodes
  parents = [0] * n
  for i in range(n):
    j = treeDict['children_left'][i]
    if j != -1:
      parents[j] = i
    k = treeDict['children_right'][i]
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

def recurse(node):
    yield node
    if len(node.getchildren()) > 0:
        for n in node.getchildren():
            for ni in recurse(n):
                yield ni

def treeToDict(bdt, tree):
  feature = []
  threshold = []
  value = []
  children_left = []
  children_right = []
  rootnode = tree[0]
  useYesNoLeaf = bool(getOptionValue(bdt, 'UseYesNoLeaf'))
  # In the fast pass add an ID
  for i, node in enumerate(recurse(rootnode)):
      node.attrib['ID'] = i
      attrib = node.attrib
      f = int(attrib['IVar']) if int(attrib['IVar']) != -1 else -2 # TMVA uses -1 for leaf, scikit-learn uses -2
      t = float(attrib['Cut'])
      vPurity = float(attrib['purity']) * float(tree.attrib['boostWeight'])
      vType = float(attrib['nType']) * float(tree.attrib['boostWeight'])
      v = vType if useYesNoLeaf else vPurity
      feature.append(f)
      threshold.append(t)
      value.append(v)

  # Now add the children left / right reference
  for i, node in enumerate(recurse(rootnode)):
      ch = node.getchildren()
      if len(ch) > 0:
          # Swap the order of the left/right child depending on cut type attribute
          if bool(int(node.attrib['cType'])):
            l = ch[0].attrib['ID']
            r = ch[1].attrib['ID']
          else:
            l = ch[1].attrib['ID']
            r = ch[0].attrib['ID']
      else:
          l = -1
          r = -1
      children_left.append(l)
      children_right.append(r)

  treeDict = {'feature' : feature, 'threshold' : threshold, 'value' : value, 'children_left' : children_left, 'children_right' : children_right}

  treeDict = addParentAndDepth(treeDict)
  return treeDict

def padTree(ensembleDict, treeDict):
  '''Pad a tree with dummy nodes if not perfectly balanced or depth < max_depth'''
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
        val = treeDict['value'][i]
        treeDict['value'].extend([val, val])
        newDepth = treeDict['depth'][i] + 1
        treeDict['depth'].extend([newDepth, newDepth])
        iRChild = len(treeDict['children_left']) - 1
        iLChild = iRChild - 1
        treeDict['children_left'][i] = iLChild
        treeDict['children_right'][i] = iRChild
    n_nodes = len(treeDict['children_left'])
  treeDict['iLeaf'] = []
  for i in range(n_nodes):
    if treeDict['depth'][i] == ensembleDict['max_depth']:
      treeDict['iLeaf'].append(i)
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
  if not os.path.isabs(yamlConfig['TMVAxml']):
    yamlConfig['sklearnPkl'] = os.path.join(configDir, yamlConfig['TMVAxml'])

  if not (yamlConfig["IOType"] == "io_parallel"):
    raise Exception('ERROR: Invalid IO type (serial not yet supported)')

  ######################
  ##  Do translation
  ######################
  if not os.path.isdir("{}/firmware".format(yamlConfig['OutputDir'])):
    os.makedirs("{}/firmware".format(yamlConfig['OutputDir']))

  xml = ET.parse(yamlConfig['TMVAxml'])
  ensembleDict = ensembleToDict(xml)
  bdt_writer(ensembleDict, yamlConfig)

if __name__ == "__main__":
  main()
