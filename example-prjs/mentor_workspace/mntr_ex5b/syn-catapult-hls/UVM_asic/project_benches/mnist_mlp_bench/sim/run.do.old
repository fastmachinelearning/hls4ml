

quietly set svLibs ""
quietly set extra_vsim_args ""
if {[info exists ::env(UVMF_EXTRA_VSIM_ARGS)]} {
  echo "Adding more args to vsim command"
  quietly set extra_vsim_args $::env(UVMF_EXTRA_VSIM_ARGS)
}
quietly set cmd [format "vsim -i -sv_seed random +UVM_TESTNAME=test_top +UVM_VERBOSITY=UVM_HIGH  -permit_unmatched_virtual_intf +notimingchecks -suppress 8887  %s %s -uvmcontrol=all -msgmode both -classdebug -assertdebug  +uvm_set_config_int=*,enable_transaction_viewing,1  -do { set NoQuitOnFinish 1; onbreak {resume}; run 0; do wave.do; set PrefSource(OpenOnBreak) 0; radix hex showbase; } optimized_debug_top_tb" $svLibs $extra_vsim_args]
eval $cmd
