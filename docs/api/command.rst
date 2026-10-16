===================================
Command Line Interface (deprecated)
===================================

The command line interface to ``hls4ml`` has been deprecated. Users are advised to use the python API. This page
documents all the commands that ``hls4ml`` supports as a reference for those that have not migrated.

----

Overview
=========

To start you can just type in ``hls4ml -h`` or ``hls4ml  --help`` in your command line, and a message like the following will show up:

.. code-block::

   usage: hls4ml [-h] [--version] {config,convert,build,report} ...

   HLS4ML - Machine learning inference in FPGAs

   positional arguments:
     {config,convert,build,report}
       config              Create a conversion configuration file
       convert             Convert Keras or ONNX model to HLS
       build               Build generated HLS project
       report              Show synthesis report of an HLS project

   optional arguments:
     -h, --help            show this help message and exit
     --version             show program's version number and exit

To get help about any particular ``command``\ , you can just do:

.. code-block::

   hls4ml command -h

For example, to get help about the ``config`` command, you can just type the following:

.. code-block::

   hls4ml config -h

----

hls4ml config
==============

.. code-block::

   hls4ml config [-h] [-m MODEL] [-w WEIGHTS] [-o OUTPUT]

This creates a conversion configuration file. Visit the :doc:`Configuration <configuration>` page for more details on how to write a configuration file.

**Arguments**


* ``-h, --help``\ : show help message and exit
* ``-m MODEL``\ , or ``--model MODEL``\ : model file to convert (we currently support Keras's ``.h5`` or ``.json`` file, ONNX's ``.onnx``\ , TensorFlow's ``pb``\ , PyTorch's ``pt``\ )
* ``-w WEIGHTS``\ , or ``--weights WEIGHTS``\ : optional weights file (if Keras's ``.json`` file is provided)
* ``-o OUTPUT``\ , or ``--output OUTPUT``\ : output file name

----

hls4ml convert
================

.. code-block::

   hls4ml convert [-h] [-c CONFIG]

Suppose you have a configuration file called ``keras-config.yml``. You can use this command with the configuration file like the following:

.. code-block::

   hls4ml convert -c keras-config.yml

**Arguments**


* ``-h, --help``\ : show help message and exit
* ``-c CONFIG``\ , or ``--config CONFIG``\ : configuration file

----

hls4ml build
==============

.. code-block::

   hls4ml build [-h] [-p PROJECT] [-c] [-s] [-r] [-v] [-e] [-l] [-a] [--reset]

Build your HLS project. If you have a project directory called ``my-hls-test``\ , you can often do the following to build the whole project with all the steps described in the arguments section:

.. code-block::

   hls4ml build -p my-hls-test -a

**Arguments**


* ``-h, --help``\ : show help message and exit.
* ``-p PROJECT``\ , or ``--project PROJECT``\ : project directory.
* ``-c, --simulation``\ : run C simulation.
* ``-s, --synthesis``\ : run C/RTL synthesis
* ``-r, --co-simulation``\ : run C/RTL co-simulation.
* ``-v, --validation``\ : run C/RTL validation.
* ``-e, --export``\ : export IP (implies -s)
* ``-l, --vivado_synthesis``\ : run Vivado synthesis (implies -s).
* ``-a, --all``\ : run C simulation, C/RTL synthesis, C/RTL co-simulation and Vivado synthesis.
* ``--reset``\ : remove any previous builds

----

hls4ml report
===============

.. code-block::

   hls4ml report [-h] [-p PROJECT] [-f]

If you have a project directory called ``my-hls-test``\ , you can get the full report about the project by doing the following:

.. code-block::

   hls4ml report -p my-hls-test -f

**Arguments**


* ``-h, --help``\ : show help message and exit.
* ``-p PROJECT``\ , or ``--project PROJECT``\ : project directory.
* ``-f, --full``\ : show full report
