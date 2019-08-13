add_files firmware/
add_files ../../bdt_utils/AddReduce.vhd
add_files ../../bdt_utils/Tree.vhd
add_files ../../bdt_utils/BDT.vhd
add_files ../../bdt_utils/Types.vhd
set_property file_type {VHDL 2008} [get_files]
# hls4ml insert synth_design
report_utilization -file util.rpt
