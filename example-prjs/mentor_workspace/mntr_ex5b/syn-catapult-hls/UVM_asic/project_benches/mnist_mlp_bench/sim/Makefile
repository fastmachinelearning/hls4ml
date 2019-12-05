
#
#----------------------------------------------------------------------
# Project         : mnist_mlp_bench Simulation Bench 
# Unit            : Bench makefile
# File            : Makefile
#----------------------------------------------------------------------
#                                          
# DESCRIPTION: This makefile includes the shared makefile and contains
#   bench level make targets.
#


# *********************************************************************************************
# UVMF library directory:
# This variable points to the UVMF release where uvmf_base_pkg directory resides.
# This variable points to release code that is not user modified.
# This variable allows for UVMF release directories to reside independent of project related verification_ip and project_benches directories.
# This code below looks "upward" for directory starting with UVMF_* and returns first match for use with the release examples.
UVMF_HOME ?= ___PLEASE_SET_AN_ENVIRONMENT_VARIABLE_NAMED_UVMF_HOME_TO_POINT_TO_THE_UVMF_INSTALLATION___

#
# Project(s) specific verification_ip library:
# Directory where reusable verification packages for interfaces, environments, utilities, etc. reside.
# This variable allows for verification_ip to reside independent of project bench and UVMF release directories.
# For examples deployed with UVMF this will be $(UVMF_HOME)/<example_group>/verification_ip
# export UVMF_VIP_LIBRARY_HOME ?= ../../../verification_ip
export UVMF_VIP_LIBRARY_HOME ?= $(PWD)/../../../verification_ip
#
# Project specific bench:
# Directory where bench specific code is located.
# This variable allows for project_benches to reside independent of verification_ip and UVMF release directories.
# For examples deployed with UVMF this will be $(UVMF_HOME)/<example_group>/project_benches/<example_bench>
# export UVMF_PROJECT_DIR ?= ../../../project_benches/mnist_mlp_bench
export UVMF_PROJECT_DIR ?= $(PWD)/..
#
#
# *********************************************************************************************

## Check PATH for required vinfo scripts
PVAL := $(shell command -v make_filelist.py 2> /dev/null)
ifndef PVAL
  MFLIST = $(UVMF_HOME)/scripts/make_filelist.py
else
  MFLIST = make_filelist.py
endif

# Set test case specific Variables
TEST_NAME            ?= test_top

TEST_SEED            ?= random
UVM_CLI_ARGS         = 

# Usage of Veloce, etc. to be input by the user (subject to defaults)
USE_VELOCE          ?= 0

# Usage of vinfo flow for generating file list
USE_VINFO           ?= 0

# Usage of Veloce and Questa profilers
USE_VELOCE_PROFILER ?= 0
USE_QUESTA_PROFILER ?= 0


# Set project Variables
TEST_PLAN_NAME       = mnist_mlp_bench_TestPlan
REPORTING_DO_FILE    = mnist_mlp_bench_reports_script

# Include makefile that includes targets for UVM_VIP_Library packages
include $(UVMF_HOME)/scripts/Makefile




# Include all requisite interface package targets for this bench
include $(MGC_HOME)/pkgs/ccs_uvmf/vip/interface_packages/ccs_pkg/Makefile

# Include all requisite environment package targets for this bench
include $(UVMF_VIP_LIBRARY_HOME)/environment_packages/mnist_mlp_env_pkg/Makefile



# Suppress novopt messages
COMMON_VSIM_ARGS      +=   -suppress 12110

# Add to default compile/load/run arguments
VCOM_ARGS             += 

# Note: vsim-3009 error can be eliminated by adding -timescale 1ps/1ps to VLOG_ARGS
VLOG_ARGS             += +define+UVM_REPORT_DISABLE_FILE_LINE
VELANALYZE_ARGS       +=
VELANALYZE_HVL_ARGS   +=

BATCH_VOPT_ARGS       +=
DEBUG_VOPT_ARGS       +=
EXTRA_VOPT_TOPS       +=

COMMON_VSIM_ARGS      +=  
COMMON_VSIM_ARGS      +=  


BATCH_VSIM_ARGS       += #-uvmcontrol=none
DEBUG_VSIM_ARGS       += 
EXTRA_VSIM_TOPS       +=  sc_main 

# Project bench package source
mnist_mlp_bench_PARAMETERS_PKG =\
$(UVMF_PROJECT_DIR)/tb/parameters/mnist_mlp_bench_parameters_pkg.sv


mnist_mlp_bench_SEQUENCES_PKG =\
$(UVMF_PROJECT_DIR)/tb/sequences/mnist_mlp_bench_sequences_pkg.sv


mnist_mlp_bench_TEST_PKG =\
$(UVMF_PROJECT_DIR)/tb/tests/mnist_mlp_bench_tests_pkg.sv

# UVMF_CHANGE_ME : Reference Verilog DUT source.
mnist_mlp_bench_VERILOG_DUT =\
/home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex5b/syn-catapult-hls/Catapult_asic/mnist_mlp.v1/concat_sim_rtl.v

# UVMF_CHANGE_ME : Reference VHDL DUT source.
mnist_mlp_bench_VHDL_DUT =\
/home/giuseppe/research/projects/fastml/hls4ml-mentor.git/example-prjs/mentor_workspace/mntr_ex5b/syn-catapult-hls/Catapult_asic/mnist_mlp.v1/concat_sim_rtl.vhdl


# Project bench package targets
COMP_mnist_mlp_bench_PARAMETERS_PKG_TGT_0 = q_comp_mnist_mlp_bench_parameters_pkg
COMP_mnist_mlp_bench_PARAMETERS_PKG_TGT_1 = v_comp_mnist_mlp_bench_parameters_pkg
COMP_mnist_mlp_bench_PARAMETERS_PKG_TGT = $(COMP_mnist_mlp_bench_PARAMETERS_PKG_TGT_$(USE_VELOCE))

comp_mnist_mlp_bench_parameters_pkg: $(COMP_mnist_mlp_bench_PARAMETERS_PKG_TGT)

q_comp_mnist_mlp_bench_parameters_pkg:
	        $(HVL_COMP_CMD) +incdir+$(UVMF_PROJECT_DIR)/tb/parameters $(mnist_mlp_bench_PARAMETERS_PKG)

v_comp_mnist_mlp_bench_parameters_pkg: q_comp_mnist_mlp_bench_parameters_pkg
	$(HDL_COMP_CMD) +incdir+$(UVMF_PROJECT_DIR)/tb/parameters $(mnist_mlp_bench_PARAMETERS_PKG)
 

comp_mnist_mlp_bench_sequence_pkg:
	$(HVL_COMP_CMD) +incdir+$(UVMF_PROJECT_DIR)/tb/sequences $(mnist_mlp_bench_SEQUENCES_PKG)

comp_mnist_mlp_bench_tests_pkg:
	$(HVL_COMP_CMD) +incdir+$(UVMF_PROJECT_DIR)/tb/tests $(mnist_mlp_bench_TEST_PKG)

# UVMF_CHANGE_ME : Add make target to compile your verilog dut here
comp_mnist_mlp_bench_verilog_dut: 
	echo "Compile your verilog DUT here"
	$(HDL_COMP_CMD) $(mnist_mlp_bench_VERILOG_DUT)

# UVMF_CHANGE_ME : Add make target to compile your vhdl dut here
comp_mnist_mlp_bench_vhdl_dut: 
	echo "Compile your vhdl DUT here"
	$(HDL_COMP_CMD_VHDL) $(mnist_mlp_bench_VHDL_DUT)

# UVMF_CHANGE_ME : Add make target to compile your dut here
comp_mnist_mlp_bench_dut: comp_mnist_mlp_bench_verilog_dut


BUILD_TGT_0 = make_build
BUILD_TGT_1 = vinfo_build
BUILD_TGT = $(BUILD_TGT_$(USE_VINFO))


comp_hvl : comp_hvl_core


comp_hvl_core :   \
	 comp_ccs_pkg  \
	comp_mnist_mlp_env_pkg \
	comp_mnist_mlp_bench_parameters_pkg comp_mnist_mlp_bench_sequence_pkg comp_mnist_mlp_bench_tests_pkg 

comp_uvmf_core : comp_uvm_pkg comp_fli_pkg comp_uvmf_base_pkg 

make_build: comp_mnist_mlp_bench_dut comp_uvmf_core comp_hvl comp_test_bench  

hvl_build:  q_comp_ccs_pkg  comp_mnist_mlp_env_pkg comp_mnist_mlp_bench_sequence_pkg comp_mnist_mlp_bench_tests_pkg hvl_comp_testbench optimize


vinfo_build: comp_mnist_mlp_bench_vhdl_dut  build_hdl_vinfo build_hvl_vinfo $(VINFO_TGT)

	$(HDL_COMP_CMD) -F hdl.vf
	$(VEL_COMP)

build: $(BUILD_TGT)




