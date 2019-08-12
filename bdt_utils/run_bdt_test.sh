mkdir msimbdtlib
vlib msimbdtlib/BDT
vmap BDT msimbdtlib/BDT

vcom -2008 -work BDT ./firmware/Constants.vhd
vcom -2008 -work BDT ../../bdt_utils/Types.vhd
vcom -2008 -work BDT ../../bdt_utils/Tree.vhd
vcom -2008 -work BDT ../../bdt_utils/AddReduce.vhd
# insert arrays
vcom -2008 -work BDT ../../bdt_utils/BDT.vhd
vcom -2008 -work BDT ./firmware/BDTTop.vhd

vlib msimbdtlib/xil_defaultlib
vmap work msimbdtlib/xil_defaultlib
vcom -2008 -work xil_defaultlib ../../bdt_utils/SimulationInput.vhd
vcom -2008 -work xil_defaultlib ../../bdt_utils/SimulationOutput.vhd
vcom -2008 -work xil_defaultlib ../../bdt_utils/BDTTestbench.vhd

vsim -batch -do test.tcl
