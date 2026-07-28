=====================
NanoXploreAccelerator
=====================

The **NanoXploreAccelerator** backend builds on the :doc:`Bambu <bambu>` backend
and turns a Bambu-generated HLS core into a complete accelerator: a float I/O
wrapper, an AXI4 slave, a PLL configuration, and a versioned ``manifest.json``
describing the result. It targets NanoXplore's `NG-ULTRA
<https://www.nanoxplore.com/>`_, a radiation-hardened FPGA with no HLS tool of
its own.

.. code-block:: python

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='NanoXploreAccelerator',
    )
    hls_model.build(synth=True, bitstream=True)

Defaults are the NG-ULTRA DevKit's: part ``nx2h540tsc`` and a 20 ns clock
period, matching the board's 50 MHz oscillator.

A deliberate seam
=================

``hls4ml`` never imports a vendor tool. The backend writes everything a
place-and-route flow needs into the project directory -- the complete RTL file
list, clock, port map and data widths, all in ``manifest.json`` and versioned so
a mismatch fails loudly -- then shells out to a single command and reads back
``report.json``:

.. code-block:: text

    hls4ml-nanoxplore-bitstream <project_dir>

The command is configurable through the ``BitStreamCommand`` config value. The
vendor-specific driver lives out of tree, which means the abstract layer can be
built and tested with no vendor licence: ``build(synth=True, bitstream=False)``
produces the wrapper, the RTL and the manifest, and stops before place and
route.

Structure
=========

``BambuAcceleratorBackend`` is abstract and unregistered. It provides the
wrapper generation, the RTL templates, the manifest and the PLL patching, and
leaves one abstract method, ``_generate_bitstream``. Other FPGA families can
reuse the layer by subclassing it.

``NanoXploreAcceleratorBackend`` is the registered concrete backend: NG-ULTRA
defaults plus the CLI call.

Clocking
========

Any requested ``ClockPeriod`` is turned into a solved ``NX_PLL_U``
configuration, spliced into the generated top level, so the hardware clock and
the timing constraint agree by construction rather than by convention. This
needs OR-Tools:

.. code-block:: bash

    pip install hls4ml[nanoxplore]
