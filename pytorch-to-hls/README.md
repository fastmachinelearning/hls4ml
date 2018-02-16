# Pytorch to HLS 

Reads Pytorch model pickle for architecture and state_dict pickle for weights and biases.

# Instructions to run

```python pytorch-to-hls.py -c pytorch-config.yml```

# Configuration

Configuration options for the HLS translation of Keras models.

*PytorchModel, PytorchDict*: For Pytorch translation, you are required to
store the model class into one pickle file, and the state_dict of the model
into another pickle file with PyTorch format. Examples are in the directory: `example-models`

*OutputDir*: Directory where your HLS project will go

*IOType*: We provide 2 options for the way inputs are input to the architecture, serially or in parallel.  The keywords are `io_serial` or `io_parallel`

*ReuseFactor*: For the running mode `io_parallel`, the calculations do not have to be fully parallelized but resources can be reused at the cost of higher latency.  A `ReuseFactor: 1` means fully parallelized and no resources are reused

*DefaultPrecision*: This is the default type of the weights, biases, accumulators, input and output vectors.  This can then be further modified by the `firmware/parameters.h` file generated in your HLS project.

# Running HLS 

```
cd my-hls-dir-test
vivado_hls -f build_prj.tcl
```
