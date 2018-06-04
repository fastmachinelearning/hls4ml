# Status

The latest stable release is **v0.1.3**, including a validated multilayer perceptron model based on our paper, [arXiv:1804.06913](https://arxiv.org/abs/1804.06913).

A beta release (**v0.1.X**) including Conv1D/2D architectures is available for testing for those interested.   

# Features

A list of suppported ML codes and architectures, including a summary table is below.  
Dependences are given in a dedicated <a href="setup/DEPENDENCIES.html">page.</a>: 

ML code support: 
   * Keras/Tensorflow, PyTorch, scikit-learn
   * Planned: xgboost 

Neural network architectures:
   * Fully Connected NNs (multi-layer perceptron) 
   * Convolutional NNs (1D/2D), in beta testing
   * Recurrent NN/LSTM, in prototyping
   * Boosted Decision Trees, in prototyping

| Architectures/Toolkits | Keras/TensorFlow | PyTorch | ScikitLearn |
|:----------:|:----------:|:----------:|:----------:|
| MLP | `supported` | `supported`| - |
| Conv1D/Conv2D | `supported` | `supported` | - |
| BDT | - | - | `in development` |
| RNN/LSTM | `in development` | - | - |
