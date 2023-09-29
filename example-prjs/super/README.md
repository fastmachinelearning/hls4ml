# Vivado AXI-master Backend and Baremetal Application

This is a workspace for testing the integration of the Vivado Accelerator AXI-master backend and baremetal applications. It is a work-in-progress repository, but it should help us to converge on a PR for the official hls4ml repository.

The _hls4ml_ fork and branch that we use in this workspace is
- ~https://github.com/GiuseppeDiGuglielmo/hls4ml/tree/gdg/axi-m~
- ~https://github.com/hls4ml-finn-mlperftiny/hls4ml/tree/fifo_depth_opt~
- https://github.com/fnal-fastml/hls4ml/tree/external-weights-vivado-accelerator

```
conda env create -f environment.yml
conda activate hls4ml-vivado-accelerator
pip install qkeras==0.9.0
pip uninstall hls4ml
pip install git+https://github.com/fnal-fastml/hls4ml.git@external-weights-vivado-accelerator#egg=hls4ml[profiling]
```

A few notes
- Please notice the changes and additions:
  - [`convert_from_keras_model`](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_backends.py#L138-L149)
  - [`write_header_file`](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_backends.py#L168)
- In particular, the function [`write_header_file`](https://github.com/hls4ml-finn-mlperftiny/hls4ml/blob/fifo_depth_opt/hls4ml/writer/vivado_accelerator_writer.py#L350-L403) in hls4ml is a draft. It should be general enough to support various models in the future.
- The [call](https://github.com/GiuseppeDiGuglielmo/test-hls4ml-backend/blob/main/test_backends.py#L168) of the same function has to be embedded in the hls4ml backend generation flow (not at the top level).

## Profile the Model
```
make run-profile
```

## Run the Hardware/Software Flow
```
make run
cd sdk
make clean sdk gui
```
At this point if you have a board locally connected you can run the software and bitstream directly from Vivado SDK.
