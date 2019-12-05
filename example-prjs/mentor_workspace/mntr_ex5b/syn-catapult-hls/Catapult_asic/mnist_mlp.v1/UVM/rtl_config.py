#! /usr/bin/env python
import uvmf_gen
# Single-file UVM configuration file for design 'mnist_mlp'
# ============================================================================
# Environment
env = uvmf_gen.EnvironmentClass('mnist_mlp')

# Current block is hierarchical. Sub-Environments:

# Define parameter for the width of each interface
env.addParamDef('input1_rsc_WIDTH','int',14112)
env.addParamDef('input1_rsc_RESET_POLARITY','bit',1)
env.addParamDef('input1_rsc_PROTOCOL_KIND','bit[1:0]',3)
env.addParamDef('output1_rsc_WIDTH','int',180)
env.addParamDef('output1_rsc_RESET_POLARITY','bit',1)
env.addParamDef('output1_rsc_PROTOCOL_KIND','bit[1:0]',3)
env.addParamDef('const_size_in_1_rsc_WIDTH','int',16)
env.addParamDef('const_size_in_1_rsc_RESET_POLARITY','bit',1)
env.addParamDef('const_size_in_1_rsc_PROTOCOL_KIND','bit[1:0]',2)
env.addParamDef('const_size_out_1_rsc_WIDTH','int',16)
env.addParamDef('const_size_out_1_rsc_RESET_POLARITY','bit',1)
env.addParamDef('const_size_out_1_rsc_PROTOCOL_KIND','bit[1:0]',2)
# Define parameter for the data width and address wisth of each external memory
# Add agents based on Interface Synthesis constraints
env.addAgent('input1_rsc','ccs','clk','rst',{'WIDTH':'14112','RESET_POLARITY':'1','PROTOCOL_KIND':'3'},'INITIATOR')
env.addAgent('output1_rsc','ccs','clk','rst',{'WIDTH':'180','RESET_POLARITY':'1','PROTOCOL_KIND':'3'},'RESPONDER')
env.addAgent('const_size_in_1_rsc','ccs','clk','rst',{'WIDTH':'16','RESET_POLARITY':'1','PROTOCOL_KIND':'2'},'RESPONDER')
env.addAgent('const_size_out_1_rsc','ccs','clk','rst',{'WIDTH':'16','RESET_POLARITY':'1','PROTOCOL_KIND':'2'},'RESPONDER')
# Add agents based on External memory

# Add TLM2 predictor
env.defineAnalysisComponent('tlm2_sysc_predictor','mnist_mlp_predictor',{'input1_rsc_ae':'ccs_transaction#(.WIDTH(input1_rsc_WIDTH),.RESET_POLARITY(input1_rsc_RESET_POLARITY),.PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND))'},{'output1_rsc_ap':'ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))','const_size_in_1_rsc_ap':'ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))','const_size_out_1_rsc_ap':'ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))'})
env.addAnalysisComponent('mnist_mlp_pred','mnist_mlp_predictor')

# Specify the scoreboards contained in this environment
env.addUvmfScoreboard('output1_rsc_sb','uvmf_catapult_scoreboard','ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))')
env.addUvmfScoreboard('const_size_in_1_rsc_sb','uvmf_catapult_scoreboard','ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))')
env.addUvmfScoreboard('const_size_out_1_rsc_sb','uvmf_catapult_scoreboard','ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))')

# Specify the connections in the environment
# Connections from the analysis port of each input agent to the input export name of the predictor
env.addConnection('input1_rsc','monitored_ap','mnist_mlp_pred','input1_rsc_ae')
env.addConnection('output1_rsc','monitored_ap','output1_rsc_sb','actual_analysis_export')
env.addConnection('mnist_mlp_pred','output1_rsc_ap','output1_rsc_sb','expected_analysis_export')
env.addConnection('const_size_in_1_rsc','monitored_ap','const_size_in_1_rsc_sb','actual_analysis_export')
env.addConnection('mnist_mlp_pred','const_size_in_1_rsc_ap','const_size_in_1_rsc_sb','expected_analysis_export')
env.addConnection('const_size_out_1_rsc','monitored_ap','const_size_out_1_rsc_sb','actual_analysis_export')
env.addConnection('mnist_mlp_pred','const_size_out_1_rsc_ap','const_size_out_1_rsc_sb','expected_analysis_export')

# Configure compilation of TLM2 SystemC code
env.addUVMCflags("-D__ASIC__ -I/opt/cad/catapult/shared/include -I../../../.. -I../../../../../inc -I../../../../../mnist_mlp/firmware -I../../../../../mnist_mlp/firmware/weights -I../../../../../mnist_mlp/firmware/nnet_utils")
# Compile TLM2 wrapper around C++ design
env.addUVMCfile("../../../../Catapult_asic/mnist_mlp.v1/UVM/mnist_mlp_tlm2_wrapper.cpp")
# Compile C++ design
env.addUVMCfile("../../../../../mnist_mlp/firmware/mnist_mlp.cpp")
# Create the environment
env.create()

# ============================================================================
# Bench
ben = uvmf_gen.BenchClass('mnist_mlp_bench','mnist_mlp')
# Define clock as clk with period=10
ben.clockHalfPeriod = '5ns'
ben.clockPhaseOffset = '0ns'
# Define reset polarity as rst with polarity=1
ben.resetAssertionLevel = True
ben.resetDuration = '25ns'

ben.addBfm('input1_rsc','ccs','clk','rst','ACTIVE',{'WIDTH':'14112','RESET_POLARITY':'1','PROTOCOL_KIND':'3'})
ben.addBfm('output1_rsc','ccs','clk','rst','ACTIVE',{'WIDTH':'180','RESET_POLARITY':'1','PROTOCOL_KIND':'3'})
ben.addBfm('const_size_in_1_rsc','ccs','clk','rst','ACTIVE',{'WIDTH':'16','RESET_POLARITY':'1','PROTOCOL_KIND':'2'})
ben.addBfm('const_size_out_1_rsc','ccs','clk','rst','ACTIVE',{'WIDTH':'16','RESET_POLARITY':'1','PROTOCOL_KIND':'2'})
ben.veloceReady = False

ben.addTopLevel('sc_main')

ben.create()
