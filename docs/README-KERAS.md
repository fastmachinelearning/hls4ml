# Keras to HLS 

Reads Keras model json for architecture and hdf5 for weights and biases.

# Instructions to run

```python keras-to-hls.py -c keras-config.yml```

# Configuration

Configuration options for the HLS translation of Keras models.

*KerasJson, KerasH5*: For Keras translation, you are required to provide `json` and `h5` model files.  
Examples are in the directory: `example-keras-model-files`

*OutputDir*: Directory where your HLS project will go

*IOType*: We provide 2 options for the way inputs are input to the architecture, serially or in parallel.  The keywords are `io_serial` or `io_parallel`

*ReuseFactor*: For the running mode `io_parallel`, the calculations do not have to be fully parallelized but resources can be reused at the cost of higher latency.  A `ReuseFactor: 1` means fully parallelized and no resources are reused

*DefaultPrecision*: This is the default type of the weights, biases, accumulators, input and output vectors.  This can then be further modified by the `firmware/parameters.h` file generated in your HLS project.

# Running HLS 

```
cd my-hls-test
vivado_hls -f build_prj.tcl
```
