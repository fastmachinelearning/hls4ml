This folder contains the implementation of custom KL divergence layer.
This is a custom implementation and not a built-in layer in any deep learning framework.
It was developed specifically for [AD@L1 CMS paper](https://www.nature.com/articles/s42256-022-00441-3) and works spesifically for the input of size `(19, 3, 1)`.

# Files

* `kl_layer.py`: contains the standalone implementation of the custom KL divergence layer
* `nnet_distance.h`: contains the HLS implementation of KL layer


# Usage

`test_extensions` function in `kl_layer.py` contains the example of how to use the KL layer.
To run do

```
python kl_layer.py
```
