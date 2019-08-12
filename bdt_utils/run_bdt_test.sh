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

vlib msimbdtlib/work
vmap work msimbdtlib/work
vcom -2008 -work work ../../bdt_utils/SimulationInput.vhd
vcom -2008 -work work ../../bdt_utils/SimulationOutput.vhd
vcom -2008 -work work ../../bdt_utils/BDTTestbench.vhd

vsim -c test -do test.tcl
