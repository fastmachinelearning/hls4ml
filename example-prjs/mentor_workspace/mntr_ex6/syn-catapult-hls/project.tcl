array set opt {
    asic       0
    csim       1
    rtlsim     1
    lsynth     1
}

#
# Reset the options to the factory defaults
#

solution new -state initial
solution options defaults

solution options set Flows/ModelSim/VLOG_OPTS {-suppress 12110}
solution options set Flows/ModelSim/VSIM_OPTS {-t ps -suppress 12110}
solution options set /Input/CppStandard c++11
solution options set /Input/TargetPlatform x86_64
solution options set /Input/CompilerFlags -DMNTR_CATAPULT_HLS
solution options set /Input/SearchPath {../inc ../keras1layer/firmware/ ../keras1layer/firmware/weights ../keras1layer/firmware/nnet_utils}

flow package require /SCVerify

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

# Add source files.
solution file add ../keras_1layer/firmware/keras1layer.cpp -type C++
solution file add ../keras_1layer/keras1layer_test.cpp -type C++ -exclude true

go new

#
#
#

go analyze

#
#
#

# Set the top module and inline all of the other functions.
#directive set -DESIGN_HIERARCHY keras1layer

# Set the top module and set FC, RELU, Sigmoid as submodules.
directive set -DESIGN_HIERARCHY { \
    keras1layer \
    {nnet::dense_large<ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, config2>} \
    {nnet::dense_large<ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, config4>} \
    {nnet::sigmoid<ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, sigmoid_config5>} \
    {nnet::relu<ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, ac_fixed<18, 8, true, AC_TRN, AC_WRAP>, relu_config3>} \
}

go compile

#
#
#

if {$opt(asic)} {
    solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
    solution library add ccs_sample_mem
} else {
    solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flva2104-2-i
    solution library add Xilinx_RAMS
    solution library add Xilinx_ROMS
    solution library add Xilinx_FIFO
}

go libraries

#
#
#

directive set -CLOCKS { \
    clk { \
        -CLOCK_PERIOD 5 \
        -CLOCK_EDGE rising \
        -CLOCK_HIGH_TIME 2.5 \
        -CLOCK_OFFSET 0.000000 \
        -CLOCK_UNCERTAINTY 0.0 \
        -RESET_KIND sync \
        -RESET_SYNC_NAME rst \
        -RESET_SYNC_ACTIVE high \
        -RESET_ASYNC_NAME arst_n \
        -RESET_ASYNC_ACTIVE low \
        -ENABLE_NAME {} \
        -ENABLE_ACTIVE high \
    } \
}

directive set /keras1layer/nnet::dense_large<input_t,layer2_t,config2> -MAP_TO_MODULE {[CCORE]}
directive set /keras1layer/nnet::sigmoid<layer4_t,result_t,sigmoid_config5> -MAP_TO_MODULE {[CCORE]}
directive set /keras1layer/nnet::dense_large<layer3_t,layer4_t,config4> -MAP_TO_MODULE {[CCORE]}
directive set /keras1layer/nnet::relu<layer2_t,layer3_t,relu_config3> -MAP_TO_MODULE {[CCORE]}

# Add debug views for CCOREs.
directive set /keras1layer/nnet::dense_large<input_t,layer2_t,config2> -CCORE_DEBUG true
directive set /keras1layer/nnet::sigmoid<layer4_t,result_t,sigmoid_config5> -CCORE_DEBUG true
directive set /keras1layer/nnet::dense_large<layer3_t,layer4_t,config4> -CCORE_DEBUG true
directive set /keras1layer/nnet::relu<layer2_t,layer3_t,relu_config3> -CCORE_DEBUG true

go assembly

#
#
#

# Top-Module I/O
directive set /keras1layer/input_1:rsc -MAP_TO_MODULE ccs_ioport.ccs_in_vld
directive set /keras1layer/layer5_out:rsc -MAP_TO_MODULE ccs_ioport.ccs_out_vld

# Arrays
directive set /keras1layer/core/nnet::dense_large<input_t,layer2_t,config2>(input_1):rsc -MAP_TO_MODULE {[Register]}
directive set /keras1layer/core/nnet::relu<layer2_t,layer3_t,relu_config3>(layer2_out):rsc -MAP_TO_MODULE {[Register]}
directive set /keras1layer/core/nnet::dense_large<layer3_t,layer4_t,config4>(layer3_out):rsc -MAP_TO_MODULE {[Register]}
directive set /keras1layer/core/nnet::sigmoid<layer4_t,result_t,sigmoid_config5>(layer4_out):rsc -MAP_TO_MODULE {[Register]}

# Loops
directive set /keras1layer/nnet::dense_large<input_t,layer2_t,config2>/core/main -UNROLL no
directive set /keras1layer/nnet::sigmoid<layer4_t,result_t,sigmoid_config5>/core/main -UNROLL no
directive set /keras1layer/nnet::dense_large<layer3_t,layer4_t,config4>/core/main -UNROLL no
directive set /keras1layer/nnet::relu<layer2_t,layer3_t,relu_config3>/core/main -UNROLL no

go architect

#
#
#

go allocate

#
# RTL
#

go extract

#
#
#

if {$opt(csim)} {
    flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
}

if {$opt(rtlsim)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
    #####flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim simgui
}

if {$opt(lsynth)} {

    if {$opt(asic)} {
        # TODO: DC is not installed yet. This will fail.
        flow run /DesignCompiler/dc_shell ./rtl.v.dc v
    } else {
        flow run /Vivado/synthesize -shell vivado/rtl.v.xv
    }

}

project save
