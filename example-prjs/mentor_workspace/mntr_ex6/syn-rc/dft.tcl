#######################################################
#######################################################

set_attribute dft_scan_map_mode tdrc_pass "$ec::DESIGN"
set_attribute dft_connect_shift_enable_during_mapping tie_off "$ec::DESIGN"
set_attribute dft_connect_scan_data_pins_during_mapping loopback "$ec::DESIGN"
set_attribute dft_mix_clock_edges_in_scan_chains true "$ec::DESIGN"

define_dft test_clock -name clk1 -domain clk1 "clk1"
define_dft test_clock -name clk2 -domain clk2 "clk2"
define_dft test_clock -name clk3 -domain clk3 "clk3"

#define_dft test_clock -name clk1 -domain <testClockDomain> -period <delay in pico sec, default 5000>  -rise <integer> -fall <integer> <portOrpin> -controllable
#set_attribute dft_dont_scan true <instance or subdesign> 
#set_attribute dft_controllable "<from pin> <inverting|non_inverting>" <to pin>
## insert_dft shadow_logic -around <instance> -mode <string> -test_mode <test_signal> 
## insert_dft test_point -location <port|pin> -test_mode <test_signal> -test_clock_pin <port|pin> -node <port|pin> -type <string>
## check_dft_rules >> ${ec::DESIGN}-tdrcs

if { $CORE_CHIP == "CHIP" } {
#	define_dft test_clock -name clk1 clk1_pad/Z -controllable 
	#define_dft test_clock -name clk1 clk1 -controllable 
	define_dft shift_enable -name SCAN_ENABLE -active high SCAN_ENABLE
	define_dft test_mode -name SCAN_MODE -active high SCAN_MODE
	define_dft test_mode -name resetB -active high resetB 
#	define_dft scan_chain -name s_chain_0 -sdi SCAN_IN -sdo SCAN_OUT -shared_output -hookup_pin_sdi SCAN_IN_pad/Z
	define_dft scan_chain -name s_chain_1 -domain clk1  -sdi SCAN_IN1 -sdo SCAN_OUT1 -shared_output 
	define_dft scan_chain -name s_chain_2 -domain clk2  -sdi SCAN_IN2 -sdo SCAN_OUT2 -shared_output 
	define_dft scan_chain -name s_chain_3 -domain clk3  -sdi SCAN_IN3 -sdo SCAN_OUT3 -shared_output 
	#define_dft scan_chain -name s_chain_0 -sdi SCAN_IN -sdo SCAN_OUT -shared_output
	set numDFTviolations [check_dft_rules]
	if {$numDFTviolations > "0"} {
		fix_dft_violations -async_set -async_reset  -test_mode SCAN_MODE -async_control SCAN_ENABLE
	}
}

if { $CORE_CHIP == "CORE" } {
	define_dft shift_enable -name SCAN_ENABLE -active high SCAN_ENABLE -create_port
	define_dft test_mode -name SCAN_MODE -active high SCAN_MODE -create_port
	define_dft scan_chain -name s_chain_0 -sdi SCAN_IN -sdo SCAN_OUT -shared_output -create_port
	set numDFTviolations [check_dft_rules]
	if {$numDFTviolations > "0"} {
		fix_dft_violations -async_set -async_reset  -test_mode resetB -async_control SCAN_ENABLE
	}
}

check_dft_rules
report dft_registers
report dft_setup 

