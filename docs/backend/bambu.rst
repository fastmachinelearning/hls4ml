=====
Bambu
=====

The ``Bambu`` backend targets `Bambu/PandA <https://panda.dei.polimi.it/>`_, an
open-source high-level synthesis compiler. It converts ``hls4ml`` models to HLS
C++ and drives Bambu to synthesizable Verilog, so no proprietary HLS tool is
required on the critical path. Both ``io_parallel`` and ``io_stream`` are
supported.

Quick start
===========

.. code-block:: python

    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Bambu')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='Bambu',
        part='xc7a100tcsg324-1',
        io_type='io_parallel',
        output_dir='my_bambu_prj',
    )

    # Compile the bridge and check numerical accuracy against Keras
    hls_model.compile()
    y = hls_model.predict(X)

    # Run Bambu: C-simulation, HLS synthesis, RTL co-simulation and (optionally)
    # a Vivado logic-synthesis pass for post-route resource/timing/power numbers.
    hls_model.build(csim=True, synth=True, cosim=True, vsynth=True)

    report = hls4ml.report.parse_bambu_report('my_bambu_prj')

``part`` must be a device Bambu knows about; the supported names are the keys of
``partname_to_bambu`` in ``hls4ml/backends/bambu/bambu_backend.py``. Requiring
``vsynth=True`` (and therefore the resource/timing/power numbers) needs Vivado on
the ``PATH``; the rest of the flow only needs the ``bambu`` executable.

Post-route utilization, timing and power numbers are parsed from the reports
Bambu's Vivado flow produces, reusing the shared report helpers in
``hls4ml/report/vivado_report.py``.

Known limitations
=================

* Large completely-partitioned arrays crash Bambu's frontend, so dense layers
  must be kept small.
* The softmax inverse lookup table is emitted as a ``constexpr`` array. When
  ``fix_softmax_table_size`` shrinks the table (i.e. when
  ``2 ** min(input_bitwidth, table_bitwidth)`` is below the default table size),
  Bambu's clang rejects the initializer at compile time. Other softmax
  configurations work; if you hit this, widen the input/table precision or take
  the argmax on the host.
* ``-m64`` combined with ``ac_channel`` crashes ``InterfaceInfer``. The default
  path avoids this by using the headers Bambu ships.

These were reported to the PandA developers and are addressed by
`PandA-bambu#396 <https://github.com/ferrandi/PandA-bambu/pull/396>`_; the
limitations above apply to current Bambu versions until that release lands.
