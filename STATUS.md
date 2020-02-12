# Status

The latest stable release is [**v0.1.6**](https://github.com/hls-fpga-machine-learning/hls4ml/releases), including a validated Boosted Decision Tree model based on our paper, [arXiv:2002.02534](https://arxiv.org/abs/2002.02534).

# Features

A list of suppported ML codes and architectures, including a summary table is below.  Dependences are given in a dedicated <a href="setup/DEPENDENCIES.html">page.</a>

ML code support: 
   * Keras/Tensorflow, PyTorch, scikit-learn
   * Planned: xgboost 

Neural network architectures:
   * Fully Connected NNs (multi-layer perceptron)
   * Boosted Decision Trees
   * Convolutional NNs (1D/2D), in beta testing
   * Recurrent NN/LSTM, in prototyping

A summary of the on-going status of the `hls4ml` tool is in the table below.

| Architectures/Toolkits | Keras/TensorFlow | PyTorch | scikit-learn |
|:----------:|:----------:|:----------:|:----------:|
| MLP | `supported` | `supported`| - |
| Conv1D/Conv2D | `supported` | `in development` | - |
| BDT | - | - | `supported` |
| RNN/LSTM | `in development` | - | - |

Other random feature notes:
   * There is a known Vivado HLS issue where the large loop unrolls create memory issues during synthesis.  We are working to solve this issue but you may see errors related to this depending on the memory of your machine.  Please feel free to email the `hls4ml` team if you have any further questions.

## Example models

We also provide and documented several example models that have been implemented in `hls4ml` in [this Github repository](https://github.com/hls-fpga-machine-learning/models).
