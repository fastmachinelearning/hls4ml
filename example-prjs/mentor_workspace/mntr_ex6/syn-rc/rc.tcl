#######################################################
#                                                     
#  script calls other scripts
#                                                     
#######################################################

if {[catch {

	source ../scripts/init.tcl
	if {$DFT == "ON"} {source ../scripts/dft.tcl}
	source ../scripts/syn.tcl
	if {$DFT == "ON"} {source ../scripts/dft2.tcl}
	source ../scripts/syn2.tcl

	#####################################################################
	# BEGIN POSTAMBLE: DO NOT EDIT

	# Write the netlist
	write -m > $ec::outDir/r2g.v

	# Write SDC file
	write_sdc > $ec::outDir/r2g.sdc

	# Write RC script file
	write_script > $ec::outDir/r2g.g

	# Write LEC file
	write_do_lec -no_exit -revised_design $ec::outDir/r2g.v  >../../lec/scripts/rtl2map.tcl

	# END POSTAMBLE
	#####################################################################


	#####################################################################
	# Noload/zero-load analysis on final result
	#####################################################################

	report timing -full


	# end timer
	puts "\nEC INFO: End at: [clock format [clock seconds] -format {%x %X}]"
	set ec::end [clock seconds]
	set ec::seconds [expr $ec::end - $ec::start]
	puts "\nEC INFO: Elapsed-time: $ec::seconds seconds\n"

	# done
	#exit

} msg]} {
	puts "\nEC ERROR: RC could not finish successfully. Force an exit now. ($msg)\n"
	#exit -822
}

