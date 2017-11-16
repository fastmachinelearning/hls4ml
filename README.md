# hls-fpga-machine-learning

A package for creating firmware implementations of machine learning algorithms using high level synthesis language (HLS).  We translate traditional machine learning package models into HLS that can be configured for your use-case!

The project is currently in development, so please let us know your experiences and if you would like new features to be added.  

contact: tbd

## dependencies

_numpy,h5py_: required for the translation of keras model files <br/>
http://www.numpy.org <br/>
http://www.h5py.org <br/>

_pyyaml_: for configuration file parsing <br/>
https://pypi.python.org/pypi/PyYAML <br/>

_Xilinx license_: required for the simulation and synthesis of the HLS

## status

*in construction* 

ML code support: 
   * Keras/Tensorflow
   * Let us know if you would support for other ML codes.  

Neural network architectures:
   * DNNs 
   * in progress: CNNs, regressions

