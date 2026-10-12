======
Libero
======

The **Libero** backend targets Microchip PolarFire and PolarFire SoC devices through
Libero SoC and SmartHLS. As introduced in PR #1240, it is a draft backend. The
supported path is Dense layers with ``io_parallel``. Dense layers have ``latency``
and ``resource`` strategies. The example below also uses ReLU; other layer and
I/O combinations need separate validation.

Select the backend with ``backend='Libero'`` in both
``config_from_pytorch_model`` and ``convert_from_pytorch_model``. The converter
accepts ``fpga_family``, ``part``, ``board``, ``clock_period``, and
``smarthls_path``. The backend defaults to ``PolarFire``, ``MPF300``,
``hw_only``, and a 5 ns clock period. These values are written to
``config.tcl``. Set ``part`` and, when needed, ``fpga_family`` and ``board``
for the actual target. If ``smarthls_path`` is omitted, ``shls`` must be on
``PATH`` so the backend can find the SmartHLS installation.

``write()`` creates a project directory with ``firmware/myproject.cpp``,
``firmware/myproject.h``, ``firmware/defines.h``,
``firmware/parameters.h``, weight files, and copied ``nnet_utils`` headers.
It also writes a testbench, a Python prediction bridge, ``config.tcl``,
``Makefile``, ``Makefile.compile``, and ``hls4ml_config.yml``. The project name
replaces ``myproject`` when one is supplied.

``compile()`` uses ``shls sw_compile`` and the generated
``Makefile.compile`` for Python prediction. ``build()`` calls SmartHLS
``sw_compile`` and ``hw`` by default, then parses the SmartHLS report.
Optional ``cosim``, ``rtl_synth``, and ``fpga`` flags request further tool
stages. These stages require a working Libero SoC and SmartHLS installation.
The generated C++ includes SmartHLS ``hls/ap_fixpt.hpp``, ``hls/ap_int.hpp``,
and ``hls/streaming.hpp`` headers, which hls4ml does not bundle for Libero.

Validated on SmartHLS 2025.2 (Windows)
=======================================

A 4-3-1 PyTorch Dense/ReLU model with ``io_parallel`` and ``ap_fixed<16,6>``
was run through the ``shls sw`` software testbench. That run exposed the
Makefile, ``uint``, and FIFO-read defects addressed here. Completion of the
testbench with these fixes remains unverified in this checkout. ``shls hw``
synthesis and co-simulation were not run.

Example: a residual-correction model for a control loop
=========================================================

This example trains a small model on synthetic residuals. It writes an HLS
project; it does not run synthesis.

.. code-block:: python

    import torch

    import hls4ml

    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    features = torch.randn(64, 16)
    residuals = 0.125 * features[:, :4].sum(dim=1, keepdim=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(features), residuals)
        loss.backward()
        optimizer.step()

    model.eval()
    config = hls4ml.utils.config_from_pytorch_model(
        model, (16,), granularity='name', backend='Libero'
    )
    for name in ('_0', '_2'):
        config['LayerName'][name]['Strategy'] = 'resource'
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        backend='Libero',
        io_type='io_parallel',
        output_dir='libero_residual_project',
    )
    hls_model.write()

The generated project is in ``libero_residual_project``. To change the target,
pass the device's ``part`` to ``convert_from_pytorch_model``. The default
``ap_fixed<16,6>`` precision quantizes weights and intermediate values;
check numerical agreement with ``predict()`` before using the project.

A worked MPFS250T example, with training under ``workers/odmr-edge-ml`` and
the measured PID plus watchdog loop under ``workers/polarfire-hybrid-control``,
is at https://github.com/jasonphillip/tychon-quantum-sensing.
