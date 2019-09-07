# Status

[Last updated: August 2, 2019]

The latest stable release is **v0.1.5**, including a validated multilayer perceptron model based on our paper, [arXiv:1804.06913](https://arxiv.org/abs/1804.06913).

A beta release including Conv1D/2D architectures is available for testing for those interested.   

# Features

A list of suppported ML codes and architectures, including a summary table is below.  Dependences are given in a dedicated <a href="setup/DEPENDENCIES.html">page.</a>

ML code support: 
   * Keras/Tensorflow, PyTorch, scikit-learn
   * Planned: xgboost 

Neural network architectures:
   * Fully Connected NNs (multi-layer perceptron) 
   * Convolutional NNs (1D/2D), in beta testing
   * Recurrent NN/LSTM, in prototyping
   * Boosted Decision Trees, in prototyping

A summary of the on-going status of the `hls4ml` tool is in the table below.

| Architectures/Toolkits | Keras/TensorFlow | PyTorch | scikit-learn |
|:----------:|:----------:|:----------:|:----------:|
| MLP | `supported` | `supported`| - |
| Conv1D/Conv2D | `supported` | `in development` | - |
| BDT | - | - | `in development` |
| RNN/LSTM | `in development` | - | - |

Other random feature notes:
   * There is a known Vivado HLS issue where the large loop unrolls create memory issues during synthesis.  We are working to solve this issue but you may see errors related to this depending on the memory of your machine.  Please feel free to email the `hls4ml` team if you have any further questions.

## Feature Documentation

-  Specific documentations of supported Keras layers and their detailed limitations can be found [here](https://github.com/hls-fpga-machine-learning/models/tree/master/keras)
