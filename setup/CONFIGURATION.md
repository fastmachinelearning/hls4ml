# Configuration

Now that you have a run a quick example workflow of hls4ml, let's go through the various configuration options that you have for the translation of your machine learning algorithm.  

One important part of hls4ml to remember is that the user is responsible for the format of the inputs.  There is no automatic formatting or normalization so this must be done in the training. 

## Keras translation

### Top level configuration

Configuration files are YAML files in hls4ml (`*.yml`). An example configuration file is [here](https://github.com/hls-fpga-machine-learning/HLS4ML/blob/v0.0.2/keras-to-hls/keras-config.yml).

It looks like this:
```
KerasJson: example-keras-model-files/KERAS_1layer.json
KerasH5:   example-keras-model-files/KERAS_1layer_weights.h5 
OutputDir: my-hls-test
ProjectName: myproject
XilinxPart: xc7vx690tffg1927-2
ClockPeriod: 5

IOType: io_parallel # options: io_serial/io_parallel
ReuseFactor: 1
DefaultPrecision: ap_fixed<18,8> 
```

There are a number of configuration options that you have.  Let's go through them.  You have basic setup parameters: 
   * **KerasJson/KerasH5**: for Keras, the model architecture and weights are stored in a `json` and `h5` file.  The path to those files are required here.
   * **OutputDir**: the output directory where you want your HLS project to appear
   * **ProjectName**: the name of the HLS project IP that is produced
   * **XilinxPart**: the particular FPGA part number that you are considering, here it's a Xilinx Virtex-7 FPGA
   * **ClockPeriod**: the clock period, in ns, at which your algorithm runs
Then you have some optimization parameters for how your algorithm runs:
   * **IOType**: your options are `io_parallel` or `io_serial` where this really defines if you are pipelining your algorithm or not
   * **ReuseFactor**: in the case that you are pipelining, this defines the pipeline interval or initiation interval
   * **DefaultPrecision**: this defines the precsion of your inputs, outputs, weights and biases.  you have a chance to further configure this more finely

For more information on the optimization parameters and what they mean, you can visit the <a href="../CONCEPTS.html">Concepts</a> chapter.

### Detailed configuration

After you create your project, you have the opportunity to do more configuration if you so choose.  
In your project, the file `<OutputDir>/firmware/<ProjectName>.cpp` is your top level file.  It has the network architecture constructed for you.  An example is [here](https://github.com/hls-fpga-machine-learning/HLS4ML/blob/v0.0.2/example-prjs/higgs-1layer/firmware/myproject.cpp) and the important snippet is:
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
An example is [here](https://github.com/hls-fpga-machine-learning/HLS4ML/blob/v0.0.2/example-prjs/higgs-1layer/firmware/parameters.h). So for example, the detailed configuration options for an example DNN layer is:
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












