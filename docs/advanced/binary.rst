================================================
Binary, Ternary and Power-of-Two Quantization
================================================

hls4ml has dedicated types and arithmetic for binary (−1 or +1) and ternary (−1, 0 or +1) data and weights, and for
power-of-two weights, as produced by QKeras quantizers and QONNX ``BipolarQuant``. This page describes how they are
implemented, which layers accept them, and what is not supported.

Types
=====

* **Binary**: the ``binary`` quantizer of QKeras, ``binary_tanh`` and QONNX ``BipolarQuant``. Binary data and
  weights are stored in one bit, with 0 standing for −1 and 1 for +1. A binary activation is stored this way when it
  follows a batch normalization (see below) or comes from QONNX. Without a batch normalization in front of it, a
  binary activation outputs −1 and +1 in a 2-bit signed type. With ``binary(use_01=True)`` the values are 0 and 1, an
  ordinary 1-bit unsigned type.
* **Ternary**: the ``ternary`` quantizer and ``ternary_tanh``. Ternary data and weights are stored as they are, in a
  2-bit signed type.
* **Power-of-two weights**: the ``quantized_po2`` quantizer. Each weight is stored as a sign bit and an exponent, and
  multiplying by it is a shift.

Batch normalization before a binary or ternary activation
=========================================================

A batch normalization followed by a binary or ternary activation is merged into one layer that compares each input
with a threshold per channel (two thresholds for ternary). The thresholds are computed for the type of the input:
binary input gets thresholds for its stored 0 and 1. A Dense layer with binary or ternary weights that reads the
output of such a merged layer gets a small integer output type; when it only feeds merged layers, its bias moves
into their thresholds.

Products
========

Layers that multiply their input with weights choose a product implementation from the two types:

* binary data and binary weights: an XNOR of the two bits; the sum counts the matching bits,
* binary data or binary weights with another type: the other value or its negation,
* ternary weights: the data, its negation or 0,
* power-of-two weights: the data shifted by the exponent, negated for negative weights.

The product of power-of-two weights holds the input shifted by any exponent the exponent field can store. To keep it
small, hls4ml narrows the exponent field to the exponents the weights actually use. Weights spread over a wide range
of exponents therefore give wider products.

Layers that accept binary data
==============================

Binary data stores −1 as 0, so only layers that know this can read it:

* **Layers that keep it binary**: they copy or select values without changing them, so their output is binary too.
  These are ``Reshape`` and ``Flatten``, ``Permute``, cropping, nearest ``UpSampling``, ``Concatenate`` of binary
  inputs, max pooling, global max pooling, and ``Maximum`` and ``Minimum`` of binary inputs.
* **Layers that multiply it**, through the products above: ``Dense``, the convolutions (including depthwise, separable
  and transposed ones), batch normalization, ``Dot``, ``EinsumDense``, ``Einsum``, and the recurrent layers.
* **Padding**: padded zeros cannot be stored in the 1-bit type, so ``ZeroPadding`` of binary data outputs −1, 0 and +1
  in a 2-bit signed type. A convolution with ``'same'`` padding and a binary input gets a ``ZeroPadding`` layer in
  front of it. The layers after the padding then read 2-bit data, which costs more logic than 1-bit data.

Any other layer that reads binary data, such as ``Add``, ``Multiply``, average pooling, an activation like ``ReLU``, or
``Concatenate`` of binary and non-binary data, would read −1 as 0. Conversion stops with an error that names the layer.
To convert such a model, move the binary activation after that layer, or use ternary data, which these layers read
correctly.

Known limitations
=================

* Average pooling with an inferred (``'auto'``) output type gets the type of its input, so the average loses its
  fractional bits. Set the output precision of the pooling layer explicitly.
* An activation quantizer without fractional bits, such as ``quantized_bits(2, 1)``, gets an integer type that neither
  rounds nor saturates, unlike QKeras. Set its precision explicitly, for example to ``fixed<2,2,RND_CONV,SAT>``.
* On Catapult, ``Maximum`` and ``Minimum`` do not compile in ``io_stream``.

Backend support
===============

* **Vivado and Vitis**: everything on this page, in both ``io_parallel`` and ``io_stream``.
* **Altera**: everything on this page, except cropping and depthwise and separable convolutions, which are not
  tested there.
* **Catapult**: Dense and convolutions, including padding, with binary, fixed-point and power-of-two weights. Layers
  that move binary data are not tested.
* **Libero**: the products are implemented as on Vivado, but not tested.
