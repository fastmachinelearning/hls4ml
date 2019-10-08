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
make hls-[ asic | fpga ]-gui
# or
make hls-[ asic | fpga ]-sh
```
This will run HLS for ASIC. The design is a simple Softmax and the synthesis should really run for few minutes.

