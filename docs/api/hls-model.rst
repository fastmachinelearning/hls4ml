================
HLS Model Class
================

This page documents our hls_model class usage. You can generate generate an hls model object from a keras model through ``hls4ml``'s API:

.. code-block:: python

   import hls4ml

   # Generate a simple configuration from keras model
   config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')

   # Convert to an hls model
   hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=config, output_dir='test_prj')

After that, you can use several methods in that object. Here is a list of all the methods:


* :ref:`write <write-method>`
* :ref:`compile <compile-method>`
* :ref:`predict <predict-method>`
* :ref:`build <build-method>`
* :ref:`trace <trace-method>`

Similar functionalities are also supported through command line interface. If you prefer using them, please refer to Command Help section.

----

.. _write-method:

``write`` method
====================

Write your keras model as a hls project to ``hls_model``\ 's ``output_dir``\ :

.. code-block:: python

   hls_model.write()

----

.. _compile-method:

``compile`` method
======================

Compile your hls project.

.. code-block:: python

   hls_model.compile()

----

.. _predict-method:

``predict`` method
======================

Similar to ``keras``\ 's predict API, you can get the predictions of ``hls_model`` just by supplying an input ``numpy`` array:

.. code-block:: python

   # Suppose that you already have input array X
   # Note that you have to do hls_model.compile() before using predict

   y = hls_model.predict(X)

This is similar to doing ``csim`` simulation, but you can get your prediction results much faster. It's very helpful when you want to quickly prototype different configurations for your model.

----

.. _build-method:

``build`` method
====================

.. code-block:: python

   hls_model.build()

   #You can also read the report of the build
   hls4ml.report.read_vivado_report('hls4ml_prj')

----

.. _trace-method:

``trace`` method
====================

The trace method is an advanced version of the ``predict`` method. It's used to trace individual outputs from each layer of the hls_model. This is useful for debugging and setting the appropriate configuration.

**Return:** A dictionary where the keys are the names of the layers, and its values are the layers's outputs.

.. code-block:: python

   predict_ouputs, trace_outputs =  hls_model.trace(X)

   #We also support a similar function for keras
   keras_trace = hls4ml.model.profiling.get_ymodel_keras(keras_model, X)
