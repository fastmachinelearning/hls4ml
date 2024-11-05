===========
Activations
===========

Most activations without extra parameters are represented with the ``Activation`` layer, and those with single parameters (leaky ReLU, thresholded ReLU, ELU) as ``ParametrizedActivation``.
``PReLU`` has its own class because it has a parameter matrix (stored as a weight). The hard (piecewise linear) sigmoid and tanh functions are implemented in a ``HardActivation`` layer,
and ``Softmax`` has its own layer class.

Softmax has four implementations that the user can choose from by setting the ``implementation`` parameter:

* **latency**:  Good latency, but somewhat high resource usage. It does not work well if there are many output classes.
* **stable**:  Slower but with better accuracy, useful in scenarios where higher accuracy is needed.
* **legacy**:  An older implementation with poor accuracy, but good performance. Usually the latency implementation is preferred.
* **argmax**:  If you don't care about normalized outputs and only care about which one has the highest value, using argmax saves a lot of resources. This sets the highest value to 1, the others to 0.
