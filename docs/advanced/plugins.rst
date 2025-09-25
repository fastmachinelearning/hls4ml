=======================================
External Backend and Writer Plugins
=======================================

Starting with this release ``hls4ml`` can discover and load backend implementations from
external Python packages. This enables specialised flows—such as the AMD AIE backend—to live in
independent projects that version and iterate at their own cadence while reusing the core
conversion infrastructure.

Discovery
=========

Plugin packages advertise themselves through the ``hls4ml.backends`` Python entry point group. Each
entry either exposes a subclass of :class:`hls4ml.backends.backend.Backend` or a callable that
receives ``register_backend`` and ``register_writer`` helpers and performs any setup that is
required. ``hls4ml`` automatically scans for these entry points during ``hls4ml.backends`` import so
third-party backends become available without additional user configuration.

In addition to entry points, modules listed in the ``HLS4ML_BACKEND_PLUGINS`` environment variable
are imported and treated as registration callables. The variable accepts an ``os.pathsep`` separated
list (``:`` on Linux/macOS or ``;`` on Windows):

.. code-block:: bash

   export HLS4ML_BACKEND_PLUGINS=aie4ml.plugin:another_pkg.hls4ml_backend

Authoring a Plugin
==================

A minimal plugin registers both a backend and an accompanying writer. The example below
shows how the ``aie4ml`` package exposes its backend via ``pyproject.toml`` and a ``register``
function:

.. code-block:: toml

   [project.entry-points."hls4ml.backends"]
   AIE = "aie4ml.plugin:register"

.. code-block:: python

   # aie4ml/plugin.py
   from aie4ml.aie_backend import AIEBackend
   from aie4ml.writer import AIEWriter

   def register(*, register_backend, register_writer):
       register_writer('AIE', AIEWriter)
       register_backend('AIE', AIEBackend)

When the plugin is installed, ``hls4ml.backends.get_available_backends()`` will report the new
backend just like the built-in FPGA toolflows.

Packaging Data Files
====================

Backends often rely on firmware templates or device description files. These assets should be
packaged alongside the Python sources using the usual ``setuptools`` mechanisms (``package-data`` or
``include-package-data``) so they are available from the installed distribution.

For an end-to-end example see the companion ``aie4ml`` [https://github.com/dimdano/aie4ml] package that ships alongside this project
as a standalone distribution; it encapsulates the existing AMD AIE backend as an installable plugin
depending on ``hls4ml``.
