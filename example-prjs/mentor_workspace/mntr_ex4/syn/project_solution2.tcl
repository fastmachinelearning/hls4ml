# Reset the options to the factory defaults
solution new -state initial
solution options defaults

solution options set Flows/ModelSim/VLOG_OPTS {-suppress 12110}
solution options set Flows/ModelSim/VSIM_OPTS {-t ps -suppress 12110}
solution options set /Input/CppStandard c++11
solution options set /Input/TargetPlatform x86_64
solution options set /Input/CompilerFlags -DMNTR_CATAPULT_HLS
solution options set /Input/SearchPath {../inc ../my-hls-test/firmware ../my-hls-test/firmware/weights ../../../nnet_utils}

flow package require /SCVerify

solution file add ../my-hls-test/myproject_test.cpp -type C++ -exclude true
solution file add ../my-hls-test/firmware/myproject.cpp -type C++

directive set -DESIGN_GOAL area
#directive set -OLD_SCHED false
directive set -SPECULATE true
directive set -MERGEABLE true
directive set -REGISTER_THRESHOLD 4096
directive set -MEM_MAP_THRESHOLD 32
directive set -LOGIC_OPT false
directive set -FSM_ENCODING none
directive set -FSM_BINARY_ENCODING_THRESHOLD 64
directive set -REG_MAX_FANOUT 0
directive set -NO_X_ASSIGNMENTS true
directive set -SAFE_FSM false
directive set -REGISTER_SHARING_MAX_WIDTH_DIFFERENCE 8
directive set -REGISTER_SHARING_LIMIT 0
directive set -ASSIGN_OVERHEAD 0
directive set -TIMING_CHECKS true
directive set -MUXPATH true
directive set -REALLOC true
directive set -UNROLL no
directive set -IO_MODE super
directive set -CHAN_IO_PROTOCOL standard
directive set -ARRAY_SIZE 1024
directive set -REGISTER_IDLE_SIGNAL false
directive set -IDLE_SIGNAL {}
directive set -STALL_FLAG false
directive set -TRANSACTION_DONE_SIGNAL true
directive set -DONE_FLAG {}
directive set -READY_FLAG {}
directive set -START_FLAG {}
directive set -BLOCK_SYNC none
directive set -TRANSACTION_SYNC ready
directive set -DATA_SYNC none
directive set -CLOCKS {clk {-CLOCK_PERIOD 0.0 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}
directive set -RESET_CLEARS_ALL_REGS true
directive set -CLOCK_OVERHEAD 20.000000
directive set -OPT_CONST_MULTS use_library
directive set -CHARACTERIZE_ROM false
directive set -PROTOTYPE_ROM true
directive set -ROM_THRESHOLD 64
directive set -CLUSTER_ADDTREE_IN_COUNT_THRESHOLD 0
directive set -CLUSTER_OPT_CONSTANT_INPUTS true
directive set -CLUSTER_RTL_SYN false
directive set -CLUSTER_FAST_MODE false
directive set -CLUSTER_TYPE combinational
directive set -COMPGRADE fast

go new

go analyze

directive set -DESIGN_HIERARCHY myproject
#directive set -DESIGN_HIERARCHY {{nnet::softmax<result_t, result_t, softmax_config4>}}

go compile

solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flva2104-2-i
solution library add Xilinx_RAMS
solution library add Xilinx_ROMS
solution library add Xilinx_FIFO

go libraries

directive set -CLOCKS {clk {-CLOCK_PERIOD 5.0 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 2.5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

go assembly

# I/O
directive set /myproject/data:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_vld
directive set /myproject/res:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_vld

# Arrays
directive set /myproject/core/layer1_out:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/logits1:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/layer2_out:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/logits2:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/layer3_out:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/logits3:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/logits4:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<input_t,layer1_t,config1>:mult:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<input_t,layer1_t,config1>:acc:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer1_t,layer2_t,config2>:mult:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer1_t,layer2_t,config2>:acc:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer2_t,layer3_t,config3>:mult:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer2_t,layer3_t,config3>:acc:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer3_t,result_t,config4>:mult:rsc -MAP_TO_MODULE {[Register]}
directive set /myproject/core/nnet::compute_layer<layer3_t,result_t,config4>:acc:rsc -MAP_TO_MODULE {[Register]}
#directive set /myproject/core/nnet::softmax<result_t,result_t,softmax_config4>:exp_res:rsc -MAP_TO_MODULE Xilinx_RAMS.BLOCK_1R1W_RBW
#directive set /myproject/core/nnet::softmax<result_t,result_t,softmax_config4>:exp_res:rsc -GEN_EXTERNAL_ENABLE true
#directive set /myproject/core/nnet::softmax<result_t,result_t,softmax_config4>:data_cache:rsc -MAP_TO_MODULE Xilinx_RAMS.BLOCK_1R1W_RBW
#directive set /myproject/core/nnet::softmax<result_t,result_t,softmax_config4>:data_cache:rsc -GEN_EXTERNAL_ENABLE true
#
## Loops
#directive set /myproject/core/main -PIPELINE_INIT_INTERVAL 1
#directive set /myproject/core/main/Product1 -UNROLL yes
#directive set /myproject/core/main/Product1/Product2 -UNROLL yes
#directive set /myproject/core/main/Accum1 -UNROLL yes
#directive set /myproject/core/main/Accum1/Accum2 -UNROLL yes
#directive set /myproject/core/main/Result -UNROLL yes
#directive set /myproject/core/main/nnet::relu<layer1_t,layer1_t,relu_config1>:for -UNROLL yes
#directive set /myproject/core/main/Product1#1 -UNROLL yes
#directive set /myproject/core/main/Product1#1/Product2#1 -UNROLL yes
#directive set /myproject/core/main/Accum1#1 -UNROLL yes
#directive set /myproject/core/main/Accum1#1/Accum2#1 -UNROLL yes
#directive set /myproject/core/main/Result#1 -UNROLL yes
#directive set /myproject/core/main/nnet::relu<layer2_t,layer2_t,relu_config2>:for -UNROLL yes
#directive set /myproject/core/main/Product1#2 -UNROLL yes
#directive set /myproject/core/main/Product1#2/Product2#2 -UNROLL yes
#directive set /myproject/core/main/Accum1#2 -UNROLL yes
#directive set /myproject/core/main/Accum1#2/Accum2#2 -UNROLL yes
#directive set /myproject/core/main/Result#2 -UNROLL yes
#directive set /myproject/core/main/nnet::relu<layer3_t,layer3_t,relu_config3>:for -UNROLL yes
#directive set /myproject/core/main/Product1#3 -UNROLL yes
#directive set /myproject/core/main/Product1#3/Product2#3 -UNROLL yes
#directive set /myproject/core/main/Accum1#3 -UNROLL yes
#directive set /myproject/core/main/Accum1#3/Accum2#3 -UNROLL yes
#directive set /myproject/core/main/Result#3 -UNROLL yes
#directive set /myproject/core/main/nnet::init_exp_table<softmax_config4,1024>:for -UNROLL yes
#directive set /myproject/core/main/nnet::init_exp_table<softmax_config4,1024>:for/div<32,30,AC_TRN,AC_WRAP,32,32,AC_TRN,AC_WRAP,32,16,AC_TRN,AC_WRAP>#1:for -UNROLL yes
#directive set /myproject/core/main/nnet::init_exp_table<softmax_config4,1024>:for/mgc_ac_hcordic<26,3,true,AC_TRN,AC_WRAP,26,3,true,AC_TRN,AC_WRAP,HCordicConstants::ROT_Z,HCordicConstants::SCALE_1>:for -UNROLL yes
#directive set /myproject/core/main/nnet::init_invert_table<softmax_config4,1024>:for -UNROLL yes
#directive set /myproject/core/main/nnet::init_invert_table<softmax_config4,1024>:for/div<32,32,AC_TRN,AC_WRAP,32,32,AC_TRN,AC_WRAP,18,8,AC_TRN,AC_WRAP>#1:for -UNROLL yes
#directive set /myproject/core/main/nnet::init_invert_table<softmax_config4,1024>:for/div<18,8,AC_TRN,AC_WRAP,18,8,AC_TRN,AC_WRAP,18,8,AC_TRN,AC_WRAP>#1:for -UNROLL yes
#directive set /myproject/core/main/nnet::softmax<result_t,result_t,softmax_config4>:for#1 -UNROLL yes
#directive set /myproject/core/main/nnet::softmax<result_t,result_t,softmax_config4>:for#1/nnet::softmax<result_t,result_t,softmax_config4>:for#1:for -UNROLL yes
#directive set /myproject/core/main/nnet::softmax<result_t,result_t,softmax_config4>:for#2 -UNROLL yes
#
#go architect
#
#directive set /myproject/core -DESIGN_GOAL Latency
#
#go extract
#
#flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
#flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
##flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim simgui
#
##flow run /Vivado/synthesize -shell vivado/rtl.v.xv
