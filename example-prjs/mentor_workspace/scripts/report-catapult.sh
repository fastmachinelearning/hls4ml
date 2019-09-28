#!/bin/bash

#
# Collect the HLS results from a Catapult HLS project.
#

if [ ! $# -eq 1 ]; then
    echo "ERROR: Usage: $0 <project-name>"
    exit 1
fi

PROJECT=$1

ARCH="v1"

PROJECT_DIR=Catapult/$PROJECT.$ARCH

CATAPULT_HLS_REPORT_FILE=$PROJECT_DIR/rtl.rpt
if [ ! -f $CATAPULT_HLS_REPORT_FILE ]; then echo "ERROR: File $CATAPULT_HLS_REPORT_FILE does not exist!"; exit 1; fi

GLOBAL_VIVADO_REPORT_XML="$PROJECT_DIR/$CATAPULT_HLS_SOLUTION/impl/report/verilog/$PROJECT"\_"export.xml"
GLOBAL_VIVADO_LOG="$PROJECT_DIR/$CATAPULT_HLS_SOLUTION/impl/report/verilog/autoimpl.log"

#
# XML parser of the Catapult HLS report.
#
function get_catapult_hls_version {
    echo $CATAPULT_HLS_VERSION
}

function get_catapult_hls_fpga_part {
    echo $CATAPULT_HLS_FPGA_PART
}

function get_catapult_hls_top_module {
    echo $CATAPULT_HLS_TOP_MODULE
}

function get_catapult_hls_target_clk {
    echo $CATAPULT_HLS_TARGET_CLK
}

function get_catapult_hls_estimated_clk {
    echo $CATAPULT_HLS_ESTIMATED_CLK
}

function get_catapult_hls_best_latency {
    local CATAPULT_HLS_REPORT_FILE=$1
    local DESIGN_TOTAL_STRING=$(grep "Design Total" $REPORT_FILE)
    local CATAPULT_HLS_LATENCY=`echo $DESIGN_TOTAL_STRING | awk '{print $4}'`
    echo $CATAPULT_HLS_LATENCY
}

function get_catapult_hls_worst_latency {
    local CATAPULT_HLS_REPORT_FILE=$1
    local DESIGN_TOTAL_STRING=$(grep "Design Total" $REPORT_FILE)
    local CATAPULT_HLS_LATENCY=`echo $DESIGN_TOTAL_STRING | awk '{print $4}'`
    echo $CATAPULT_HLS_LATENCY
}

function get_catapult_hls_min_ii {
    echo $CATAPULT_HLS_MIN_II
}

function get_catapult_hls_max_ii {
    echo $CATAPULT_HLS_MAX_II
}

function get_catapult_hls_bram {
    echo $CATAPULT_HLS_BRAM
}

function get_catapult_hls_dsp {
    echo $CATAPULT_HLS_DSP
}

function get_catapult_hls_ff {
    echo $CATAPULT_HLS_FF
}

function get_catapult_hls_lut {
    echo $CATAPULT_HLS_LUT
}

function get_catapult_hls_available_bram {
    echo $CATAPULT_HLS_AVAILABLE_BRAM
}

function get_catapult_hls_available_dsp {
    echo $CATAPULT_HLS_AVAILABLE_DSP
}

function get_catapult_hls_available_ff {
    echo $CATAPULT_HLS_AVAILABLE_FF
}

function get_catapult_hls_available_lut {
    echo $CATAPULT_HLS_AVAILABLE_LUT
}

#
# XML parser of the Vivado report.
#

function get_vivado_achieved_clk {
    local VIVADO_REPORT_XML=$1
    local VIVADO_ACHIEVED_CLK=$(xmllint --xpath "/profile/TimingReport/AchievedClockPeriod/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_ACHIEVED_CLK=?; fi
    echo $VIVADO_ACHIEVED_CLK
}

function get_vivado_bram {
    local VIVADO_REPORT_XML=$1
    local VIVADO_BRAM=$(xmllint --xpath "/profile/AreaReport/Resources/BRAM/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_BRAM=?; fi
    echo $VIVADO_BRAM
}

function get_vivado_dsp {
    local VIVADO_REPORT_XML=$1
    local VIVADO_DSP=$(xmllint --xpath "/profile/AreaReport/Resources/DSP/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_DSP=?; fi
    echo $VIVADO_DSP
}

function get_vivado_ff {
    local VIVADO_REPORT_XML=$1
    local VIVADO_FF=$(xmllint --xpath "/profile/AreaReport/Resources/FF/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_FF=?; fi
    echo $VIVADO_FF
}

function get_vivado_lut {
    local VIVADO_REPORT_XML=$1
    local VIVADO_LUT=$(xmllint --xpath "/profile/AreaReport/Resources/LUT/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_LUT=?; fi
    echo $VIVADO_LUT
}

#
# Misc.
#

function get_git_revision {
    local GIT_REVISION=$(git rev-parse --short HEAD)
    if [ ! $? -eq 0 ]; then GIT_REVISION=?; fi
    echo $GIT_REVISION
}

#
# AWK on Vivado HLS / Vivado logs
#

function get_total_execution_time {
    echo $TOTAL_EXECUTION_TIME
}

function get_catapult_hls_execution_time {
    echo $CATAPULT_HLS_EXECUTION_TIME
}

function get_vivado_execution_time {
    echo $VIVADO_EXECUTION_TIME
}

function get_rtl_simulation_execution_time {
    echo $RTL_SIMULATION_EXECUTION_TIME
}

function get_catapult_hls_exit_value {
    echo $CATAPULT_HLS_EXIT_VAL
}

function get_vivado_exit_value {
    echo $VIVADO_EXIT_VAL
}

function get_rtl_simulation_exit_value {
    echo $RTL_SIMULATION_EXIT_VAL
}

CATAPULT_HLS_VERSION=$(get_catapult_hls_version $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_FPGA_PART=$(get_catapult_hls_fpga_part $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_TOP_MODULE=$(get_catapult_hls_top_module $GLOBAL_CATAPULT_HLS_REPORT_XML)

GIT_REVISION=$(get_git_revision)

CATAPULT_HLS_TARGET_CLK=$(get_catapult_hls_target_clk $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_ESTIMATED_CLK=$(get_catapult_hls_estimated_clk $GLOBAL_CATAPULT_HLS_REPORT_XML)
VIVADO_ACHIEVED_CLK=$(get_vivado_achieved_clk $GLOBAL_VIVADO_REPORT_XML)

CATAPULT_HLS_BEST_LATENCY=$(get_catapult_hls_best_latency $GLOBAL_CATAPULT_HLS_REPORT_FILE)
CATAPULT_HLS_WORST_LATENCY=$(get_catapult_hls_worst_latency $GLOBAL_CATAPULT_HLS_REPORT_FILE)
CATAPULT_HLS_MIN_II=$(get_catapult_hls_min_ii $GLOBAL_CATAPULT_HLS_REPORT_FILE)
CATAPULT_HLS_MAX_II=$(get_catapult_hls_max_ii $GLOBAL_CATAPULT_HLS_REPORT_FILE)

CATAPULT_HLS_BRAM=$(get_catapult_hls_bram $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_DSP=$(get_catapult_hls_dsp $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_FF=$(get_catapult_hls_ff $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_LUT=$(get_catapult_hls_lut $GLOBAL_CATAPULT_HLS_REPORT_XML)

CATAPULT_HLS_AVAILABLE_BRAM=$(get_catapult_hls_available_bram $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_AVAILABLE_DSP=$(get_catapult_hls_available_dsp $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_AVAILABLE_FF=$(get_catapult_hls_available_ff $GLOBAL_CATAPULT_HLS_REPORT_XML)
CATAPULT_HLS_AVAILABLE_LUT=$(get_catapult_hls_available_lut $GLOBAL_CATAPULT_HLS_REPORT_XML)

VIVADO_BRAM=$(get_vivado_bram $GLOBAL_VIVADO_REPORT_XML)
VIVADO_DSP=$(get_vivado_dsp $GLOBAL_VIVADO_REPORT_XML)
VIVADO_FF=$(get_vivado_ff $GLOBAL_VIVADO_REPORT_XML)
VIVADO_LUT=$(get_vivado_lut $GLOBAL_VIVADO_REPORT_XML)

TOTAL_EXECUTION_TIME=$(get_total_execution_time $GLOBAL_CATAPULT_HLS_LOG)
CATAPULT_HLS_EXECUTION_TIME=$(get_catapult_hls_execution_time $GLOBAL_CATAPULT_HLS_LOG)
VIVADO_EXECUTION_TIME=$(get_vivado_execution_time $GLOBAL_CATAPULT_HLS_LOG)
RTL_SIMULATION_EXECUTION_TIME=$(get_rtl_simulation_execution_time $GLOBAL_CATAPULT_HLS_LOG)

CATAPULT_HLS_EXIT_VAL=$(get_catapult_hls_exit_value $GLOBAL_CATAPULT_HLS_REPORT_XML)
VIVADO_EXIT_VAL=$(get_vivado_exit_value $VIVADO_REPORT_XML)
RTL_SIMULATION_EXIT_VAL=$(get_rtl_simulation_exit_value $GLOBAL_CATAPULT_HLS_LOG)

clear
CATAPULT_HLS_BRAM_PERC=`bc -l <<< "scale=2; 100 * $CATAPULT_HLS_BRAM / $CATAPULT_HLS_AVAILABLE_BRAM"`
CATAPULT_HLS_FF_PERC=`bc -l <<< "scale=2; 100 * $CATAPULT_HLS_FF / $CATAPULT_HLS_AVAILABLE_FF"`
CATAPULT_HLS_DSP_PERC=`bc -l <<< "scale=2; 100 * $CATAPULT_HLS_DSP / $CATAPULT_HLS_AVAILABLE_DSP"`
CATAPULT_HLS_LUT_PERC=`bc -l <<< "scale=2; 100 * $CATAPULT_HLS_LUT / $CATAPULT_HLS_AVAILABLE_LUT"`
VIVADO_BRAM_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_BRAM / $CATAPULT_HLS_AVAILABLE_BRAM"`
VIVADO_FF_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_FF / $CATAPULT_HLS_AVAILABLE_FF"`
VIVADO_DSP_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_DSP / $CATAPULT_HLS_AVAILABLE_DSP"`
VIVADO_LUT_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_BRAM / $CATAPULT_HLS_AVAILABLE_BRAM"`

printf "INFO: Mentor Catapult HLS Report\n"
printf "INFO: === Info ================================================================\n"
printf "INFO: Project: %-20s | %-20s | %-20s\n" "Dir: $PROJECT_DIR" "Top: $CATAPULT_HLS_TOP_MODULE" "Arch: $CATAPULT_HLS_SOLUTION"
printf "INFO: Vivado : %-20s | %-20s\n" "Ver: $CATAPULT_HLS_VERSION" "Part: $CATAPULT_HLS_FPGA_PART"
printf "INFO: Git    : $GIT_REVISION\n"
printf "INFO: === Execution ===========================================================\n"
printf "INFO: Time (sec): Total: $TOTAL_EXECUTION_TIME\n"
printf "INFO: Time (sec): %-20s | %-20s | %-20s\n" "HLS: $CATAPULT_HLS_EXECUTION_TIME" "LS: $VIVADO_EXECUTION_TIME" "RTL-sim: $RTL_SIMULATION_EXECUTION_TIME"
printf "INFO: Exit VAL  : %-20s | %-20s | %-20s\n" "HLS: $CATAPULT_HLS_EXIT_VAL" "LS: $VIVADO_EXIT_VAL" "RTL-sim: $RTL_SIMULATION_EXIT_VAL"
printf "INFO: === Timing ==============================================================\n"
printf "INFO: CLK (ns) : %-20s | %-20s | %-20s\n" "Target   : $CATAPULT_HLS_TARGET_CLK" "HLS: $CATAPULT_HLS_ESTIMATED_CLK" "LS: $VIVADO_ACHIEVED_CLK"
printf "INFO: LAT (clk): %-20s | %-20s\n" "Best: $CATAPULT_HLS_BEST_LATENCY" "Worst: $CATAPULT_HLS_WORST_LATENCY"
printf "INFO: II (clk) : %-20s | %-20s\n" "Min: $CATAPULT_HLS_MIN_II" "Max: $CATAPULT_HLS_MAX_II"
printf "INFO: === Resources ===========================================================\n"
printf "INFO: BRAM : %-20s | %-20s | %-20s\n" "AVBL: $CATAPULT_HLS_AVAILABLE_BRAM" "HLS: $CATAPULT_HLS_BRAM ($CATAPULT_HLS_BRAM_PERC%)" "LS: $VIVADO_BRAM ($VIVADO_BRAM_PERC%)"
printf "INFO: DSP  : %-20s | %-20s | %-20s\n" "AVBL: $CATAPULT_HLS_AVAILABLE_DSP" "HLS: $CATAPULT_HLS_DSP ($CATAPULT_HLS_DSP_PERC%)" "LS: $VIVADO_DSP ($VIVADO_DSP_PERC%)"
printf "INFO: FF   : %-20s | %-20s | %-20s\n" "AVBL: $CATAPULT_HLS_AVAILABLE_FF" "HLS: $CATAPULT_HLS_FF ($CATAPULT_HLS_FF_PERC%)" "LS: $VIVADO_FF ($VIVADO_FF_PERC%)"
printf "INFO: LUT  : %-20s | %-20s | %-20s\n" "AVBL: $CATAPULT_HLS_AVAILABLE_LUT" "HLS: $CATAPULT_HLS_LUT ($CATAPULT_HLS_LUT_PERC%)" "LS: $VIVADO_LUT ($VIVADO_LUT_PERC%)"
printf "INFO: =========================================================================\n"

