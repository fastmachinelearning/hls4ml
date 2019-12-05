# MNIST Classifier - On-Chip Weights 

## Directory Structure

- `doc` documentation
- `inc` header files to port Vivado HLS fixed-point arithmetic to Catapult HLS
- `keras-config.yml` hls4ml configuration file
- `mnist_mlp` hls4ml project directory (some files are manually edited to support Catapult HLS)
- `rtl-tb` RTL testbench
- `sim` simulation w/out licenses
- `syn-catapult-hls` Catapult HLS synthesis scripts
- `syn-rc` RTL Compiler synthesis scripts
- `syn-vivado-hls` Vivado HLS synthesis scripts
- `uvm` UVM project (from Catapult HLS, in alpha stage)

## Simulation (w/out licenses)

You still need Catapult HLS and Vivado HLS installed on your local machine. Update the scripts `sim/envsetup-catapult-nolicense.sh` and `sim/envsetup-vivado-nolicense.sh` to match the installation paths on your local machine.

Use a new console for Catapult HLS and Vivado HLS:
```
cd sim
source envsetup-catapult-nolicense.sh
# or
# source envsetup-vivado-nolicense.sh
make clean
make run-catapult
# or
# make run-vivado
```

## Vivado HLS Synthesis

You need Vivado HLS installed on your local machine. Update the script `syn-vivado-hls/envsetup-vivado.sh` to match the installation paths on your local machine. This script exports the environment variable for Vivado HLS.

Use a new console.
```
cd syn-vivado-hls
source envsetup-vivado.sh
make hls-sh
# or for more targets
# make <TAB>
```

Edit `syn-vivado-hls/build_prj.tcl` to configure the Vivado HLS project.


## Catapult HLS Synthesis

You need Catapult HLS installed on your local machine. Update the script `syn-catapult-hls/envsetup-catapult.sh` to match the installation paths on your local machine. This script exports the environment variable for Catapult HLS.

Use a new console.
```
cd syn-catapult-hls
source envsetup-catapult.sh
```

You can target either FPGA or ASIC technology.

- To target FPGA:
  ```
  make ultraclean
  make hls-fpga-sh
  # or for the GUI mode
  # make hls-fpga-gui
  # or for more targets
  # make <TAB>
  ```
  The generated Verilog RTL code is `syn-catapult-hls/Catapult_fpga/mnist_mlp.v1/concat_rtl.v`.

- To target ASIC:
  ```
  make ultraclean
  make hls-asic-sh
  # or for the GUI mode
  # make hls-asic-gui
  # or for more targets
  # make <TAB>
  ```
  The generated Verilog RTL code is `syn-catapult-hls/Catapult_asic/mnist_mlp.v1/concat_rtl.v`. In the same folder, there are few additional files you may interested in.

## RTL Simulation w/ a Verilog Testbench

You can find a simple testbench for `syn-catapult-hls/Catapult_asic/mnist_mlp.v1/concat_rtl.v` in is provided in `rtl-tb/tb/mnist_mlp_tb.v`.

To run the simulation with [Icarus Verilog](http://iverilog.icarus.com) and show the waveform with [GTKWave](http://gtkwave.sourceforge.net):
```
cd rtl-tb/sim
make show
```

To run the simulation in Modelsim
```
cd rtl-tb/msim
make run
# Use run -all in the Modelsim console
```

## RTL Compiler Synthesis

You can run RTL Compiler (`rc`) in the directory `syn-rc`. There are two set of synthesis scripts:
- From Fermi Lab, see `syn-rc/scripts/fermi_lab`
- From Catapult HLS, see `syn-rc/script/catapult_hls`

You need RTL Compiler installed on your local machine. Update the script `syn-rc/envsetup-rc.sh` to match the installation paths on your local machine. This script exports the environment variable for RTL Compiler.

Use a new console.

```
cd syn-rc
source envsetup-rc.sh 
```

To run RTL compiler with the Catapul-HLS generated scripts:
```
make ultraclean
make run-rc-catapult-hls
```

To run RTL compiler with the Fermi Lab scripts:
```
make ultraclean
make run-rc-catapult-hls
```

