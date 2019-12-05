 

onerror resume
wave update off

wave spacer -backgroundcolor Salmon { input1_rsc }
wave add uvm_pkg::uvm_phase::m_run_phases...uvm_test_top.environment.agent_inst_name.agent_inst_name_monitor.txn_stream -radix string
wave group input1_rsc_bus
wave add -group input1_rsc_bus hdl_top.input1_rsc_bus.* -radix hexadecimal
wave insertion [expr [wave index insertpoint] +1]
wave spacer -backgroundcolor Salmon { output1_rsc }
wave add uvm_pkg::uvm_phase::m_run_phases...uvm_test_top.environment.agent_inst_name.agent_inst_name_monitor.txn_stream -radix string
wave group output1_rsc_bus
wave add -group output1_rsc_bus hdl_top.output1_rsc_bus.* -radix hexadecimal
wave insertion [expr [wave index insertpoint] +1]
wave spacer -backgroundcolor Salmon { const_size_in_1_rsc }
wave add uvm_pkg::uvm_phase::m_run_phases...uvm_test_top.environment.agent_inst_name.agent_inst_name_monitor.txn_stream -radix string
wave group const_size_in_1_rsc_bus
wave add -group const_size_in_1_rsc_bus hdl_top.const_size_in_1_rsc_bus.* -radix hexadecimal
wave insertion [expr [wave index insertpoint] +1]
wave spacer -backgroundcolor Salmon { const_size_out_1_rsc }
wave add uvm_pkg::uvm_phase::m_run_phases...uvm_test_top.environment.agent_inst_name.agent_inst_name_monitor.txn_stream -radix string
wave group const_size_out_1_rsc_bus
wave add -group const_size_out_1_rsc_bus hdl_top.const_size_out_1_rsc_bus.* -radix hexadecimal
wave insertion [expr [wave index insertpoint] +1]

wave update on
WaveSetStreamView

