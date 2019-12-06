# mnist_mlp environment packages source and make target

# Include all requisite sub-environment package targets for this bench

mnist_mlp_ENV_PKG =\
		 $(UVMF_VIP_LIBRARY_HOME)/environment_packages/mnist_mlp_env_pkg/mnist_mlp_env_pkg.sv

comp_mnist_mlp_env_pkg:  compile_sc  
		$(HVL_COMP_CMD) +incdir+$(UVMF_VIP_LIBRARY_HOME)/environment_packages/mnist_mlp_env_pkg $(mnist_mlp_ENV_PKG)

# UVMConnect support - target is 'compile_uvmc'.
include $(UVMF_HOME)/common/uvmc/Makefile
compile_sc: compile_uvmc
	-@echo "Compiling SystemC TLM models"
	sccom $(MACHINE_ARCH) -I$(QUESTA_HOME)/verilog_src/uvmc-$(UVMC_VER)/src/connect/sc -I$(QUESTA_HOME)/includes/systemc -g -Wall -fmessage-length=0 -fPIC -pipe -scv -DQUESTA -DSC_INCLUDE_DYNAMIC_PROCESSES -O0 -DDEBUG_COUTX -DSC_INCLUDE_MTI_AC -DMTI_BIND_SC_MEMBER_FUNCTION -verbose -D__ASIC__ -I/opt/cad/catapult/shared/include -I../../../.. -I../../../../../inc -I../../../../../mnist_mlp/firmware -I../../../../../mnist_mlp/firmware/weights -I../../../../../mnist_mlp/firmware/nnet_utils ../../../../Catapult_asic/mnist_mlp.v1/UVM/mnist_mlp_tlm2_wrapper.cpp
	sccom $(MACHINE_ARCH) -I$(QUESTA_HOME)/verilog_src/uvmc-$(UVMC_VER)/src/connect/sc -I$(QUESTA_HOME)/includes/systemc -g -Wall -fmessage-length=0 -fPIC -pipe -scv -DQUESTA -DSC_INCLUDE_DYNAMIC_PROCESSES -O0 -DDEBUG_COUTX -DSC_INCLUDE_MTI_AC -DMTI_BIND_SC_MEMBER_FUNCTION -verbose -D__ASIC__ -I/opt/cad/catapult/shared/include -I../../../.. -I../../../../../inc -I../../../../../mnist_mlp/firmware -I../../../../../mnist_mlp/firmware/weights -I../../../../../mnist_mlp/firmware/nnet_utils ../../../../../mnist_mlp/firmware/mnist_mlp.cpp
	sccom $(MACHINE_ARCH) -g -link -I$(QUESTA_HOME)/verilog_src/uvmc-$(UVMC_VER)/src/connect/sc -I$(QUESTA_HOME)/includes/systemc -g -Wall -fmessage-length=0 -fPIC -pipe -work work -scv -DQUESTA -DSC_INCLUDE_DYNAMIC_PROCESSES -O0
