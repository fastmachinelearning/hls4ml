Frequently asked questions
==========================

**What is hls4ml?**

``hls4ml`` is a tool for converting neural network models into FPGA firmware. hls4ml is aimed at low-latency applications, such as triggering at the Large Hadron Collider (LHC) at CERN, but is applicable to other domains requiring microsecond latency. See the full documentation for more details.

**How does hls4ml work?**

``hls4ml`` takes the models from Keras, PyTorch and ONNX (optionally quantized with the respective quantization libraries) and produces high-level synthesis code (based on C++) that can be converted to FPGA firmware using the HLS compilers from different vendors (AMD/Xilinx, Intel/Altera, Catapult...).

**How is hls4ml so fast?**

``hls4ml`` stores all weights on-chip for fast access and has tuneable parallelism. As a consequence, the size of the model that can be successfully converted into firmware with hls4ml largely depends on the amount of available resources on the target FPGA. Therefore it is highly recommended to compress the model with quantization (via QKeras or HGQ for Keras or Brevitas for PyTorch) and pruning. Additionally, ``hls4ml`` exploits the parallelism available in an FPGA or ASIC by implementing a spatial dataflow architecture.

**Will my model work with hls4ml?**

``hls4ml`` supports many common layers found in MLP, CNN and RNN architectures, however some seldom-used features of these layers may not be supported. Novel architectures such as graph networks or transformers are in various stages of development and are currently not stable for end-users. See the status and features page for more information. Models with custom layers can be supported through extension API. If you encounter a feature not yet supported, open a new issue.

**Will my model with X parameters fit an FPGA model Y?**

It depends. ``hls4ml`` has been successfully used with quantized models with `O` (10k) parameters, while for some architectures going beyond `O` (1000) parameters is not doable even on the largest FPGAs. The number of parameters of a model is generally not a good estimate of the performance on an FPGA as the computational complexity of different types of NN layers has big effects on the resource consumption on an FPGA. For example, a CNN or GNN may reuse the same parameter in many operations. Furthermore, model compression in the form of quantization and pruning can significantly change the footprint of the model on the FPGA. For these reasons, we discourage the use of this metric for estimating performance.

If you're looking for a quick estimate of the resource usage and latency for a given model without synthesis, look into `rule4ml <https://github.com/IMPETUS-UdeS/rule4ml>`_ and `wa-hls4ml <https://github.com/Dendendelen/wa-hls4ml>`_ projects.

LLMs and large vision transformers are not supported nor planned.

**How do I get started with hls4ml?**

We strongly recommend interested users unfamiliar with FPGAs or model compression techniques to review the `hls4ml tutorials <https://github.com/fastmachinelearning/hls4ml-tutorial>`_ to get an overview of the features and conversion workflow.

**How do I contribute to hls4ml development?**

We're always welcoming new contributions. If you have an interesting feature in mind feel free to start a new discussion thread with your proposal. We also have regular meetings online to discuss the status of developments where you can be invited to present your work. To receive announcements, `request to be added to our CERN e-group <https://e-groups.cern.ch/e-groups/Egroup.do?egroupName=hls-fml>`_. Furthermore, check the `CONTRIBUTING <https://github.com/fastmachinelearning/hls4ml/blob/main/CONTRIBUTING.md>`_ document for a set of technical requirements for making contributions to the hls4ml project.


Common HLS synthesis issues
***************************

**Stop unrolling loop ... because it may cause large runtime and excessive memory usage due to increase in code size.**

This error is common with models that are too large to fit on the FPGA given the ``IOType`` used. If you are using ``io_parallel``, consider switching to ``io_stream``, which prevents unrolling all arrays. It may help to also use the ``Resource`` strategy. Pruning or quantizing the model may not help as it is related to the size of the loops. If possible, try to reduce the number of neurons/filters of your model to reduce the size of the activation tensors and thus number of iterations of loops.

**cannot open shared object file ...: No such file or directory.**

This is usually an indication that the compilation failed due to incorrect HLS code being produced. It is most likely a bug in hls4ml. Please open a bug report. Note that the displayed error message may be the same but the cause can be different. Unless you're sure that the existing bug reports show the same underlying issue, it is better to open a separate bug report.

**My hls4ml predictions don't match the original Keras/PyTorch/ONNX ones**

``hls4ml`` uses fixed-point precision types to represent internal data structures, unlike the floating-point precision types used for computation in upstream ML toolkits. If the used bit width is not sufficiently wide, you may encounter issues with computation accuracy that propagates through the layers. This is especially true for models that are not fully quantized, or models with insufficient ``accum_t`` bitwidth. Look into automatic precision inference and profiling tools to resolve the issue.

Note that bit-exact behavior is not always possible, as many math functions (used by activation functions) are approximated with lookup tables.
