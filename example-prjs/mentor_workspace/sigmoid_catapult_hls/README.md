# Hello Catapult HLS

Use this example to check your environment configuration.

## Environment Setup

I prepared the environment setup script `syn/envsetup.sh` that you may edit to your convenience.

In particular, you should set:
- `PATH_TO_YOUR_CAD_TOOLS` on your local workstation. You definitely need Catapult HLS, you may need Vivado and/or Design Compiler for logic synthesis.
- `PORT` and `SERVER_NAME` to fetch your licenses.

## Run Catapult HLS
```
cd syn
source envsetup.sh
make hls-gui
# or
make hls-sh
```
This will run HLS for ASIC. The design is a simple Sigmoid and the synthesis should really run for few minutes.


At the top of `syn/build_prj.tcl` you can set
```
    asic       0
```
to run HLS for FPGA.
