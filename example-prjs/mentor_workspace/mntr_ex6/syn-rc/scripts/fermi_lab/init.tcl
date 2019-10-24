#######################################################
#######################################################
#
# private namespace "ec" to prevent name clash
#
namespace eval ec {}

# start timer
puts "Start at: [clock format [clock seconds] -format {%x %X}]"
set ec::start [clock seconds]

#####################################################################
# Setup file, directories, and variables
#####################################################################

#set ec::inDir           ../verilog/syn-catapult-hls-keras1layer/Catapult/keras1layer.v1
#set ec::outDir          ../output
#set ec::reportDir       ../report
set ec::inDir           ./input
set ec::outDir          ./output/fermi_lab
set ec::reportDir       ./report/fermi_lab

set ec::SYN_EFFORT      medium
set ec::MAP_EFFORT      high
set ec::INCR_EFFORT     high

#set TSMC_PDK "/asic/cad/Library/tsmc/TSMC65_VCAD/Base_PDK/V1.7A_1/1p9m6x1z1u"
set OPEN45NM_PDK "/opt/cad/catapult/pkgs/siflibs/nangate"

#set_attr library /opt/cad/catapult/pkgs/siflibs/nangate/nangate45nm_nldm.lib
#set_attr lef_library /opt/cad/catapult/pkgs/siflibs/nangate/nangate45nm.lef


set CORE_CHIP 	CHIP
set DFT OFF
#set DESIGN keras1layer

#set ec::RTL_PATH        ../verilog/syn-catapult-hls-keras1layer/Catapult/keras1layer.v1
#set ec::LIB_PATH        "$TSMC_PDK"
set ec::RTL_PATH        ./input
set ec::LIB_PATH        "$OPEN45NM_PDK"


#set ec::LIBRARY         "$TSMC_PDK/../../digital/Front_End/timing_power_noise/NLDM/tcbn65lp_200a/tcbn65lpwc.lib \
						 $TSMC_PDK/../../digital/Front_End/timing_power_noise/NLDM/tpdn65lpnv2od3_200a/tpdn65lpnv2od3wc.lib \
						 $TSMC_PDK/../../digital/Front_End/timing_power_noise/NLDM/tpan65lpnv2od3_200a/tpan65lpnv2od3wc.lib"
						 
#set ec::LIBRARY_7THVT   "$TSMC_PDK/../../digital/Front_End/timing_power_noise/NLDM/tcbn65lpbwp7thvt_141a/tcbn65lpbwp7thvtwc.lib \
						 $TSMC_PDK/../../digital/Front_End/timing_power_noise/NLDM/tpdn65lpnv2od3_200a/tpdn65lpnv2od3wc.lib"
set ec::LIBRARY          "$OPEN45NM_PDK/nangate45nm_nldm.lib"

#set ec::VERILOG_LIST    { dense.v }
set ec::VERILOG_LIST    { concat_rtl.v }



set ec::VERILOG_VERSION 2001
set ec::VHDL_LIST       {}
set ec::VHDL_VERSION    1993

#set ec::LEFLIB "/homedir/bonacini/TSMC65/Libraries/tcbn65lp_200b/TSMCHOME/digital/Back_End/lef/tcbn65lp_200a/lef/tcbn65lp_6lmT2.lef "
#set ec::LEFLIB " $TSMC_PDK/../../digital/Back_End/lef/tcbn65lp_200a/lef/tcbn65lp_6lmT1.lef \
				 $TSMC_PDK/../../digital/Back_End/lef/tpdn65lpnv2od3_140b/mt_2/6lm/lef/tpdn65lpnv2od3_6lm.lef \
				 $TSMC_PDK/../../digital/Back_End/lef/tcbn65lpbwp7thvt_141a/lef/tcbn65lpbwp7thvt_6lmT1.lef "
				 
#set ec::CAPTABLE "/homedir/bonacini/TSMC65/Libraries/tcbn65lp_200b/TSMCHOME/digital/Back_End/lef/tcbn65lp_200a/techfiles/captable/cln65lp_1p06m+alrdl_top2_rcworst.captable"
#set ec::CAPTABLE "$TSMC_PDK/../../digital/Back_End/lef/tcbn65lp_200a/techfiles/captable/cln65lp_1p06m+alrdl_top1_cworst.captable"

set ec::LEFLIB          "$OPEN45NM_PDK/nangate45nm.lef" 

#set ec::SDC             ../sdc/constraint.sdc
set ec::SDC             ./scripts/constraint.sdc


set ec::SUPPRESS_MSG    {LBR-30 LBR-31 VLOGPT-35}


# include needed script
include load_etc.tcl

#####################################################################
# Preset global variables and attributes
#####################################################################
#set_attribute super_thread_servers {localhost localhost localhost localhost} /

# define diagnostic variables
set iopt_stats 1
set global_map_report 1
set map_fancy_names 1
set path_disable_priority 0
# set report_unfolding 1
# set cost_grp_details_in_iopt 1

# define QoR related variables
# set global_area 2  ; # valid range: 0 through 6
# set dont_downsize_components 1
# set map_slackq 0
# set final_remaps 0
# set initial_target 0
# set crr 1
set crr_max_tries 300  ; # higher the number, more iterations: not much runtime penalty

# define tool setup and compatibility
set_attribute information_level 9 /  ; # valid range: 1 (least verbose) through 9 (most verbose)
set_attribute hdl_max_loop_limit 1024 /
set_attribute hdl_reg_naming_style %s_reg%s /
set_attribute gen_module_prefix G2C_DP_ /
# set_attribute endpoint_slack_opto 1 /
#set_attribute optimize_constant_0_flops false /
#set_attribute optimize_constant_1_flops false /
set_attribute input_pragma_keyword {cadence synopsys get2chip g2c} /
set_attribute synthesis_off_command translate_off /
set_attribute synthesis_on_command translate_on /
set_attribute input_case_cover_pragma {full_case} /
set_attribute input_case_decode_pragma {parallel_case} /
set_attribute input_synchro_reset_pragma sync_set_reset /
set_attribute input_synchro_reset_blk_pragma sync_set_reset_local /
set_attribute input_asynchro_reset_pragma async_set_reset /
set_attribute input_asynchro_reset_blk_pragma async_set_reset_local /
#set_attribute delayed_pragma_commands_interpreter dc /
set_attribute script_begin dc_script_begin /
set_attribute script_end dc_script_end /
set_attribute iopt_force_constant_removal true /

# triplication TMR persistence:
#set_attribute merge_combinational_hier_instance false / 

# suppress messages
suppress_messages $ec::SUPPRESS_MSG

# setup shrink_factor attribute
set_attribute shrink_factor 1.0 /

#####################################################################
# RTL and libraries setup
#####################################################################

# search paths
set_attribute hdl_search_path $ec::RTL_PATH /
set_attribute lib_search_path $ec::LIB_PATH /

# timing libraries
#create_library_domain {sc9t sc7thvt}
#set_attribute library $ec::LIBRARY sc9t
#set_attribute library $ec::LIBRARY_7THVT sc7thvt
set_attribute library $ec::LIBRARY

#set_attribute default true sc9t
#set_attribute default true sc7thvt

# lef & captbl
set_attribute lef_library $ec::LEFLIB /
#set_attribute cap_table_file $ec::CAPTABLE /

set_attribute interconnect_mode ple / 


# report time and memory
puts "\nEC INFO: Total cpu-time and memory after SETUP: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

### Power root attributes
set_attribute lp_insert_clock_gating false /
set_attribute lp_clock_gating_prefix lpg /
set_attribute lp_insert_operand_isolation true /
set_attribute hdl_track_filename_row_col true /

## Power root attributes -NEW
#set_attribute lp_insert_clock_gating true /
##set_attribute lp_clock_gating_prefix <string> /
##set_attribute lp_insert_operand_isolation true /
##set_attribute lp_operand_isolation_prefix <string> /
##set_attribute lp_power_analysis_effort <high> /
##set_attribute lp_power_unit mW /
##set_attribute lp_toggle_rate_unit /ns /
#set_attribute hdl_track_filename_row_col true /
#set_attribute lp_multi_vt_optimization_effort low /


#####################################################################
# Load RTL
#####################################################################
set_attribute auto_ungroup none /
set_attribute hdl_language sv /
set_attribute hdl_infer_unresolved_from_logic_abstract true
read_hdl $ec::VERILOG_LIST

# report time and memory
puts "\nEC INFO: Total cpu-time and memory after LOAD: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

#####################################################################
# Elaborate
#####################################################################

elaborate

# report time and memory
puts "\nEC INFO: Total cpu-time and memory after ELAB: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

#####################################################################
# Constraint setup
#####################################################################

# read sdc constraint
foreach ec::FILE_NAME $ec::SDC {
  read_sdc $ec::FILE_NAME
}

# report time and memory
puts "\nEC INFO: Total cpu-time and memory after CONSTRAINT: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

#####################################################################
# Define cost groups (clock-clock, clock-output, input-clock, input-output)
#####################################################################

define_cost_group -name I2C
define_cost_group -name C2O
define_cost_group -name I2O
define_cost_group -name C2C
path_group -from [all des seqs] -to [all des seqs] -group C2C -name C2C
path_group -from [all des seqs] -to [all des outs] -group C2O -name C2O
path_group -from [all des inps] -to [all des seqs] -group I2C -name I2C
path_group -from [all des inps] -to [all des outs] -group I2O -name I2O

#####################################################################
# Initial reports
#####################################################################

# print out the exceptions
set ec::XCEP [find /designs* -exception *]
puts "\nEC INFO: Total numbers of exceptions: [llength $ec::XCEP]\n"
catch {open $ec::reportDir/exception.all "w"} ec::FXCEP
puts $ec::FXCEP "Total numbers of exceptions: [llength $ec::XCEP]\n"
foreach ec::X $ec::XCEP {
  puts $ec::FXCEP $ec::X
}
close $ec::FXCEP

# report time and memory
puts "\nEC INFO: Total cpu-time and memory after POST-SDC: [get_attr runtime /] sec., [get_attr memory_usage /] MBytes.\n"

# report initial design
report design > $ec::reportDir/init.design

# report initial gates
report gates > $ec::reportDir/init.gate

# report initial area
report area > $ec::reportDir/init.area

# report initial timing
report timing -full > $ec::reportDir/init.timing

# report initial timing groups
report timing -end -slack 0 > $ec::reportDir/init.timing.ep
report timing -from [dc::all_inputs] > $ec::reportDir/init.timing.in
report timing -to   [dc::all_outputs] > $ec::reportDir/init.timing.out
set ec::CNT 1
foreach ec::CLK [find /designs* -clock *] {
  exec echo "####################" > $ec::reportDir/init.timing.clk$ec::CNT
  exec echo "# from clock: $ec::CLK" >> $ec::reportDir/init.timing.clk$ec::CNT
  exec echo "# to clock: $ec::CLK" >> $ec::reportDir/init.timing.clk$ec::CNT
  exec echo "####################" >> $ec::reportDir/init.timing.clk$ec::CNT
  report timing -from $ec::CLK -to $ec::CLK >> $ec::reportDir/init.timing.clk$ec::CNT
  incr ec::CNT
}

# report initial summary
puts "\nEC INFO: Reporting Initial QoR below...\n"
redirect -tee $ec::reportDir/init.qor {report qor}
puts "\nEC INFO: Reporting Initial Summary below...\n"
redirect -tee $ec::reportDir/init.summary {report summary}

report timing -lint
################################################
# DFT
################################################
set ec::DESIGN [find * -design *]

# this line is only an example on how to set part of the design on 7-tracks
#set_attribute library_domain sc7thvt CSR
##set_attribute library_domain sc7thvt core_digital/slave/slave*/synch*
