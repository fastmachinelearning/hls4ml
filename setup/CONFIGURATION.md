# Configuration
---
Now that you have a run a quick example workflow of hls4ml, let's go through the various configuration options that you have for the translation of your machine learning algorithm.  

One important part of hls4ml to remember is that the user is responsible for the format of the inputs.  There is no automatic formatting or normalization so this must be done in the training. 

---
## Keras translation

### Top level configuration

Configuration files are YAML files in hls4ml (`*.yml`). An example configuration file is [here](https://github.com/hls-fpga-machine-learning/hls4ml/blob/master/example-models/keras-config.yml).

It looks like this:

```
KerasJson: keras/KERAS_3layer.json
KerasH5:   keras/KERAS_3layer_weights.h5 #You can also use h5 file from Keras's model.save() and leave KerasJson blank.
InputData: keras/KERAS_3layer_input_features.dat
OutputPredictions: keras/KERAS_3layer_predictions.dat
OutputDir: my-hls-test
ProjectName: myproject
XilinxPart: xcku115-flvb2104-2-i
ClockPeriod: 5

IOType: io_parallel # options: io_serial/io_parallel
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
    Strategy: Latency 
  LayerType:
    Dense:
      ReuseFactor: 2
      Strategy: Resource
      Compression: True
```

There are a number of configuration options that you have.  Let's go through them.  You have basic setup parameters: 
   * **KerasJson/KerasH5**: for Keras, the model architecture and weights are stored in a `json` and `h5` file.  The path to those files are required here. 
   We also support keras model's file obtained just from `model.save()`. In this case you can just leave the `KerasJson:` field blank. 
   * **InputData/OutputPredictions**: path to your input/predictions of the model. If none is supplied, then hls4ml will create aritificial data for simulation. The data used above in the example can be found [here](https://cernbox.cern.ch/index.php/s/2LTJVVwCYFfkg59). We also support `npy` data files. We welcome suggestions on more input data types to support. 
   * **OutputDir**: the output directory where you want your HLS project to appear
   * **ProjectName**: the name of the HLS project IP that is produced
   * **XilinxPart**: the particular FPGA part number that you are considering, here it's a Xilinx Virtex-7 FPGA
   * **ClockPeriod**: the clock period, in ns, at which your algorithm runs
Then you have some optimization parameters for how your algorithm runs:
   * **IOType**: your options are `io_parallel` or `io_serial` where this really defines if you are pipelining your algorithm or not
   * **ReuseFactor**: in the case that you are pipelining, this defines the pipeline interval or initiation interval
   * **Strategy**: Optimization strategy on FPGA, either "Latency" or "Resource". If none is supplied then hl4ml uses "Latency" as default. Note that a reuse factor larger than 1 should be specified when using "resource" strategy. An example of using larger reuse factor can be found [here.](https://github.com/hls-fpga-machine-learning/models/tree/master/keras/KERAS_dense)
   * **Precision**: this defines the precsion of your inputs, outputs, weights and biases. It is denoted by `ap_fixed<X,Y>`, where `Y` is the number of bits representing the signed number above the binary point (i.e. the integer part), and `X` is the total number of bits.
   Additionally, integers in fixed precision data type (`ap_int<N>`, where `N` is a bit-size from 1 to 1024) can also be used. You have a chance to further configure this more finely with per-layer configuration described below.


**Per-layer configuration:**
In the hls4ml configuration file, it is possible to specify the model *Precision* and *ReuseFactor* with finer granularity.

Under the `HLSConfig` heading, these can be set for the `Model`, per `LayerType`, per `LayerName`, and for named variables within the layer (for precision only). The most basic configuration may look like this:

```
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
```
This configuration use `ap_fixed<16,6>` for every variable and a ReuseFactor of 1 throughout.

Specify all `Dense` layers to use a different precision like this:

```
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
  LayerType:
    Dense:
      Precision: ap_fixed<14,5>
```

In this case, all variables in any `Dense` layers will be represented with `ap_fixed<14,5>` while any other layer types will use `ap_fixed<16,6>`.

A specific layer can be targeted like this:
```
 HLSConfig:
    Model:
      Precision: ap_fixed<16,6>
      ReuseFactor: 16
    LayerName:
      dense1:
        Precision: 
          weight: ap_fixed<14,2>
          bias: ap_fixed<14,4>
          result: ap_fixed<16,6>
        ReuseFactor: 12
        Strategy: Resource
```

In this case, the default model configuration will use `ap_fixed<16,6>` and a `ReuseFactor` of 16. The layer named `dense1` (defined in the user provided model architecture file) will instead use different precision for the `weight`, `bias`, and `result` (output) variables, a `ReuseFactor` of 12, and the `Resource` strategy (while the model default is `Latency` strategy.

More than one layer can have a configuration specified, e.g.:
```
HLSConfig:
  Model:
   ...
  LayerName:
    dense1:
       ...
    batchnormalization1:
       ...
    dense2:
       ...
```

For more information on the optimization parameters and what they mean, you can visit the <a href="../CONCEPTS.html">Concepts</a> chapter.

---

### Detailed configuration

After you create your project, you have the opportunity to do more configuration if you so choose.  
In your project, the file `<OutputDir>/firmware/<ProjectName>.cpp` is your top level file.  It has the network architecture constructed for you.  An example is [here](https://github.com/hls-fpga-machine-learning/hls4ml/blob/master/example-prjs/higgs-1layer/firmware/myproject.cpp) and the important snippet is:

```
layer1_t layer1_out[N_LAYER_1];
#pragma HLS ARRAY_PARTITION variable=layer1_out complete
layer1_t logits1[N_LAYER_1];
#pragma HLS ARRAY_PARTITION variable=logits1 complete
nnet::compute_layer<input_t, layer1_t, config1>(data, logits1, w1, b1);
nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

result_t logits2[N_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=logits2 complete
nnet::compute_layer<layer1_t, result_t, config2>(layer1_out, logits2, w2, b2);
nnet::sigmoid<result_t, result_t, sigmoid_config2>(logits2, res);
```

You can see, for the simple 1-layer DNN, the computation (`nnet::compute_layer`) and activation (`nnet::relu`/`nnet::sigmoid`) caluclation for each layer.  For each layer, it has its own additional configuration parameters, e.g. `config1`.

In your project, the file `<OutputDir>/firmware/parameters.h` stores all the configuration options for each neural network library.
An example is [here](https://github.com/hls-fpga-machine-learning/hls4ml/blob/master/example-prjs/higgs-1layer/firmware/parameters.h). So for example, the detailed configuration options for an example DNN layer is:
```
struct config1 : nnet::layer_config {
        static const unsigned n_in = N_INPUTS;
        static const unsigned n_out = N_LAYER_1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
```
It is at this stage that a user can even further configure their network HLS implementation in finer detail.












