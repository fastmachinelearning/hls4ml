======================================
PQuantML
======================================

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://www.apache.org/licenses/LICENSE-2.0
.. image:: https://github.com/nroope/PQuant/actions/workflows/python-publish.yml/badge.svg
   :target: https://pquantml.readthedocs.io
.. image:: https://badge.fury.io/py/pquant-ml.svg
   :target: https://badge.fury.io/py/pquant-ml

PQuantML is a hardware-aware model compression framework supporting:
  - Joint pruning + quantization
  - Layer-wise precision configuration
  - Flexible training pipelines
  - PyTorch and Keras V3 implementations
  - Integration with hardware-friendly toolchains (e.g., hls4ml)

PQuantML enables efficient deployment of compact neural networks on resource-constrained hardware such as FPGAs and embedded accelerators.


Key Features
------------

  - **Joint Quantization + Pruning**: Combine bit-width reduction with structured pruning.
  - **Flexible Precision Control**: Per-layer and mixed-precision configuration.
  - **Hardware-Aware Objective**: Include resource constraints (DSP, LUT, BRAM) in training.
  - **Simple API**: Configure compression through a single YAML or Python object.
  - **PyTorch Integration**: Works with custom training/validation loops.
  - **Export Support**: Model conversion towards hardware toolchains.


.. code-block:: python
   :caption: Simple example

   import torch
   from pquant import dst_config
   from pquant.layers import PQDense
   from pquant.activations import PQActivation

   # Define the compression config and model
   config = dst_config()
   config.training_parameters.epochs = 1000
   config.quantization_parameters.default_data_integer_bit = 3.
   config.quantization_parameters.default_data_fractional_bits = 2.
   config.quantization_parameters.default_weight_fractional_bits = 3.
   config.quantization_parameters.use_relu_multiplier = False

   def build_model(config):
      class Model(torch.nn.Module):
        def __init__(self):
          super().__init__()
          self.dense1 = PQDense(config, 16, 64,
                                    in_quant_bits = (1, 3, 3))
          self.relu1 = PQActivation(config, "relu")
          self.dense2 = PQDense(config, 64, 32)
          self.relu2 = PQActivation(config, "relu")
          self.dense3 = PQDense(config, 32, 32)
          self.relu3 = PQActivation(config, "relu")
          self.dense4 = PQDense(config, 32, 5,
                                quantize_output=True,
                                out_quant_bits=(1, 3, 3))

        def forward(self, x):
          x = self.relu1(self.dense1(x))
          x = self.relu2(self.dense2(x))
          x = self.relu3(self.dense3(x))
          x = self.dense4(x)
          return x

      return Model(config)

   PQmodel = build_model(config)
   PQmodel(torch.rand((1, 16)))

   ... # Training, evaluation, and anything else you want to do with the model

   hls_config = config_from_pytorch_model(
      PQmodel,
      input_shape=input_shape,
      )
   hls_model = convert_from_pytorch_model(PQmodel, ...)
   # Model-wise precision propagation is done automatically for PQuantML models for bit-exactness
   # Do NOT pass precision config if you don't know what you are doing

   hls_model.compile()

.. note::
   Do not pass any precision configuration from ``hls4ml.converters.convert_from_<frontend>_model`` in general. PQuantML-defined models will invoke model-wise precision propagation automatically to ensure bit-exactness between the PQuantML model and the generated HLS code (See `here <./precision.html>`__ for more details).
