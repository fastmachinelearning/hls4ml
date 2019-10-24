#######################################################
#######################################################
### Optional additional DFT commands.
#set_compatible_test_clocks -all
connect_scan_chains -auto_create_chains
report dft_chains > $ec::reportDir/dft_chains.rpt
## report dft_setup > ${ec::DESIGN}-DFTsetup
write_scandef > ../output/scan.def
## write_atpg [-stil_or_mentor_or_cadence] > ${ec::DESIGN}-ATPG
## write_dft_abstract_model > ${ec::DESIGN}-scanAbstract
## write_hdl -abstract > ${ec::DESIGN}-logicAbstract
