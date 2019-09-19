#!/bin/bash

#
# Collect the HLS results from a Vivado HLS project.
#

if [ ! $# -eq 1 ]; then
    echo "ERROR: Usage: $0 <project-name>"
    exit 1
fi

PROJECT=$1

PROJECT_DIR=$PROJECT\_prj
CSV_FILE=$PROJECT.csv

APP_FILE=$PROJECT_DIR/vivado_hls.app
if [ ! -f $APP_FILE ]; then echo "ERROR: File $APP_FILE does not exist!"; exit 1; fi

VIVADO_HLS_SOLUTION="solution1"

GLOBAL_VIVADO_HLS_REPORT_XML="$PROJECT_DIR/$VIVADO_HLS_SOLUTION/syn/report/csynth.xml"
GLOBAL_VIVADO_HLS_LOG="$PROJECT_DIR/../vivado_hls.log"

GLOBAL_VIVADO_REPORT_XML="$PROJECT_DIR/$VIVADO_HLS_SOLUTION/impl/report/verilog/$PROJECT"\_"export.xml"
GLOBAL_VIVADO_LOG="$PROJECT_DIR/$VIVADO_HLS_SOLUTION/impl/report/verilog/autoimpl.log"

#
# XML parser of the Vivado HLS report.
#
function get_vivado_hls_version {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_VERSION=$(xmllint --xpath "/profile/ReportVersion/Version/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_VERSION=?; fi
    echo $VIVADO_HLS_VERSION
}

function get_vivado_hls_fpga_part {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_FPGA_PART=$(xmllint --xpath "/profile/UserAssignments/Part/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_FPGA_PART=?; fi
    echo $VIVADO_HLS_FPGA_PART
}

function get_vivado_hls_top_module {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_TOP_MODULE=$(xmllint --xpath "/profile/UserAssignments/TopModelName/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_TOP_MODULE=?; fi
    echo $VIVADO_HLS_TOP_MODULE
}

function get_vivado_hls_target_clk {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_TARGET_CLK=$(xmllint --xpath "/profile/UserAssignments/TargetClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_TARGET_CLK=?; fi
    echo $VIVADO_HLS_TARGET_CLK
}

function get_vivado_hls_estimated_clk {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_ESTIMATED_CLK=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_ESTIMATED_CLK=?; fi
    echo $VIVADO_HLS_ESTIMATED_CLK
}

function get_vivado_hls_best_latency {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_BEST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_BEST_LATENCY=?; fi
    echo $VIVADO_HLS_BEST_LATENCY
}

function get_vivado_hls_worst_latency {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_WORST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_WORST_LATENCY=?; fi
    echo $VIVADO_HLS_WORST_LATENCY
}

function get_vivado_hls_min_ii {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_MIN_II=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-min/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_MIN_II=?; fi
    echo $VIVADO_HLS_MIN_II
}

function get_vivado_hls_max_ii {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_MAX_II=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-max/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_MAX_II=?; fi
    echo $VIVADO_HLS_MAX_II
}

function get_vivado_hls_bram {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_BRAM=$(xmllint --xpath "/profile/AreaEstimates/Resources/BRAM_18K/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_BRAM=?; fi
    echo $VIVADO_HLS_BRAM
}

function get_vivado_hls_dsp {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_DSP=$(xmllint --xpath "/profile/AreaEstimates/Resources/DSP48E/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_DSP=?; fi
    echo $VIVADO_HLS_DSP
}

function get_vivado_hls_ff {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_FF=$(xmllint --xpath "/profile/AreaEstimates/Resources/FF/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_FF=?; fi
    echo $VIVADO_HLS_FF
}

function get_vivado_hls_lut {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_LUT=$(xmllint --xpath "/profile/AreaEstimates/Resources/LUT/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_LUT=?; fi
    echo $VIVADO_HLS_LUT
}

function get_vivado_hls_available_bram {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_AVAILABLE_BRAM=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/BRAM_18K/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_AVAILABLE_BRAM=?; fi
    echo $VIVADO_HLS_AVAILABLE_BRAM
}

function get_vivado_hls_available_dsp {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_AVAILABLE_DSP=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/DSP48E/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_AVAILABLE_DSP=?; fi
    echo $VIVADO_HLS_AVAILABLE_DSP
}

function get_vivado_hls_available_ff {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_AVAILABLE_FF=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/FF/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_AVAILABLE_FF=?; fi
    echo $VIVADO_HLS_AVAILABLE_FF
}

function get_vivado_hls_available_lut {
    local VIVADO_HLS_REPORT_XML=$1
    local VIVADO_HLS_AVAILABLE_LUT=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/LUT/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_AVAILABLE_LUT=?; fi
    echo $VIVADO_HLS_AVAILABLE_LUT
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
    local VIVADO_HLS_LOG=$1
    local TOTAL_EXECUTION_TIME=$(grep "Total elapsed time:" $VIVADO_HLS_LOG | awk '{print $7}')
    if [ -z "$TOTAL_EXECUTION_TIME" ]; then TOTAL_EXECUTION_TIME=?; fi
    echo $TOTAL_EXECUTION_TIME
}

function get_vivado_hls_execution_time {
    local VIVADO_HLS_LOG=$1
    local VIVADO_HLS_EXECUTION_TIME_STRING=$(grep "C/RTL SYNTHESIS COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
    if [ -z "$VIVADO_HLS_EXECUTION_TIME_STRING" ]; then
        VIVADO_HLS_EXECUTION_TIME=?;
    else
        VIVADO_HLS_EXECUTION_TIME=$(echo $VIVADO_HLS_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    fi
    echo $VIVADO_HLS_EXECUTION_TIME
}

function get_vivado_execution_time {
    local VIVADO_HLS_LOG=$1
    local VIVADO_EXECUTION_TIME_STRING=$(grep "EXPORT IP COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
    if [ -z "$VIVADO_EXECUTION_TIME_STRING" ]; then
        VIVADO_EXECUTION_TIME=?;
    else
        VIVADO_EXECUTION_TIME=$(echo $VIVADO_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    fi
    echo $VIVADO_EXECUTION_TIME
}

function get_rtl_simulation_execution_time {
    local VIVADO_HLS_LOG=$1
    local RTL_SIMULATION_EXECUTION_TIME_STRING=$(grep "C/RTL SIMULATION COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
    if [ -z "$RTL_SIMULATION_EXECUTION_TIME_STRING" ]; then
        RTL_SIMULATION_EXECUTION_TIME=?;
    else
        RTL_SIMULATION_EXECUTION_TIME=$(echo $RTL_SIMULATION_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    fi
    echo $RTL_SIMULATION_EXECUTION_TIME
}

function get_vivado_hls_exit_value {
    local VIVADO_HLS_REPORT_XML=$1
    if [ -f "$VIVADO_HLS_REPORT_XML" ]; then
        local VIVADO_HLS_EXIT_VAL=0
    else
        local VIVADO_HLS_EXIT_VAL=1
    fi
    echo $VIVADO_HLS_EXIT_VAL
}

function get_vivado_exit_value {
    local VIVADO_REPORT_XML=$1
    if [ -f "$VIVADO_REPORT_XML" ]; then
        VIVADO_EXIT_VAL=0
    else
        VIVADO_EXIT_VAL=1
    fi
    echo $VIVADO_EXIT_VAL
}

function get_rtl_simulation_exit_value {
    local VIVADO_HLS_LOG=$1
    local SIM_RESULTS=$(grep "C/RTL co-simulation finished: PASS" $VIVADO_HLS_LOG | wc -l)
    if [ $SIM_RESULTS == 1 ]; then # Simulation Passed
        local RTL_SIMULATION_EXIT_VAL=0
    else
        local RTL_SIMULATION_EXIT_VAL=1
    fi
    echo $RTL_SIMULATION_EXIT_VAL
}

VIVADO_HLS_VERSION=$(get_vivado_hls_version $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_FPGA_PART=$(get_vivado_hls_fpga_part $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_TOP_MODULE=$(get_vivado_hls_top_module $GLOBAL_VIVADO_HLS_REPORT_XML)

GIT_REVISION=$(get_git_revision)

VIVADO_HLS_TARGET_CLK=$(get_vivado_hls_target_clk $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_ESTIMATED_CLK=$(get_vivado_hls_estimated_clk $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_ACHIEVED_CLK=$(get_vivado_achieved_clk $GLOBAL_VIVADO_REPORT_XML)

VIVADO_HLS_BEST_LATENCY=$(get_vivado_hls_best_latency $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_WORST_LATENCY=$(get_vivado_hls_worst_latency $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_MIN_II=$(get_vivado_hls_min_ii $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_MAX_II=$(get_vivado_hls_max_ii $GLOBAL_VIVADO_HLS_REPORT_XML)

VIVADO_HLS_BRAM=$(get_vivado_hls_bram $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_DSP=$(get_vivado_hls_dsp $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_FF=$(get_vivado_hls_ff $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_LUT=$(get_vivado_hls_lut $GLOBAL_VIVADO_HLS_REPORT_XML)

VIVADO_HLS_AVAILABLE_BRAM=$(get_vivado_hls_available_bram $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_AVAILABLE_DSP=$(get_vivado_hls_available_dsp $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_AVAILABLE_FF=$(get_vivado_hls_available_ff $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_HLS_AVAILABLE_LUT=$(get_vivado_hls_available_lut $GLOBAL_VIVADO_HLS_REPORT_XML)

VIVADO_BRAM=$(get_vivado_bram $GLOBAL_VIVADO_REPORT_XML)
VIVADO_DSP=$(get_vivado_dsp $GLOBAL_VIVADO_REPORT_XML)
VIVADO_FF=$(get_vivado_ff $GLOBAL_VIVADO_REPORT_XML)
VIVADO_LUT=$(get_vivado_lut $GLOBAL_VIVADO_REPORT_XML)

TOTAL_EXECUTION_TIME=$(get_total_execution_time $GLOBAL_VIVADO_HLS_LOG)
VIVADO_HLS_EXECUTION_TIME=$(get_vivado_hls_execution_time $GLOBAL_VIVADO_HLS_LOG)
VIVADO_EXECUTION_TIME=$(get_vivado_execution_time $GLOBAL_VIVADO_HLS_LOG)
RTL_SIMULATION_EXECUTION_TIME=$(get_rtl_simulation_execution_time $GLOBAL_VIVADO_HLS_LOG)

VIVADO_HLS_EXIT_VAL=$(get_vivado_hls_exit_value $GLOBAL_VIVADO_HLS_REPORT_XML)
VIVADO_EXIT_VAL=$(get_vivado_exit_value $VIVADO_REPORT_XML)
RTL_SIMULATION_EXIT_VAL=$(get_rtl_simulation_exit_value $GLOBAL_VIVADO_HLS_LOG)

VIVADO_HLS_BRAM_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_HLS_BRAM / $VIVADO_HLS_AVAILABLE_BRAM"`
VIVADO_HLS_FF_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_HLS_FF / $VIVADO_HLS_AVAILABLE_FF"`
VIVADO_HLS_DSP_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_HLS_DSP / $VIVADO_HLS_AVAILABLE_DSP"`
VIVADO_HLS_LUT_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_HLS_LUT / $VIVADO_HLS_AVAILABLE_LUT"`
VIVADO_BRAM_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_BRAM / $VIVADO_HLS_AVAILABLE_BRAM"`
VIVADO_FF_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_FF / $VIVADO_HLS_AVAILABLE_FF"`
VIVADO_DSP_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_DSP / $VIVADO_HLS_AVAILABLE_DSP"`
VIVADO_LUT_PERC=`bc -l <<< "scale=2; 100 * $VIVADO_BRAM / $VIVADO_HLS_AVAILABLE_BRAM"`

printf "INFO: Xilinx Vivado Report\n"
printf "INFO: === Info ================================================================\n"
printf "INFO: Project: %-20s | %-20s | %-20s\n" "Dir: $PROJECT_DIR" "Top: $VIVADO_HLS_TOP_MODULE" "Arch: $VIVADO_HLS_SOLUTION"
printf "INFO: Vivado : %-20s | %-20s\n" "Ver: $VIVADO_HLS_VERSION" "Part: $VIVADO_HLS_FPGA_PART"
printf "INFO: Git    : $GIT_REVISION\n"
printf "INFO: === Execution ===========================================================\n"
printf "INFO: Time (sec): Total: $TOTAL_EXECUTION_TIME\n"
printf "INFO: Time (sec): %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXECUTION_TIME" "LS: $VIVADO_EXECUTION_TIME" "RTL-sim: $RTL_SIMULATION_EXECUTION_TIME"
printf "INFO: Exit VAL  : %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXIT_VAL" "LS: $VIVADO_EXIT_VAL" "RTL-sim: $RTL_SIMULATION_EXIT_VAL"
printf "INFO: === Timing ==============================================================\n"
printf "INFO: CLK (ns) : %-20s | %-20s | %-20s\n" "Target   : $VIVADO_HLS_TARGET_CLK" "HLS: $VIVADO_HLS_ESTIMATED_CLK" "LS: $VIVADO_ACHIEVED_CLK"
printf "INFO: LAT (clk): %-20s | %-20s\n" "Best: $VIVADO_HLS_BEST_LATENCY" "Worst: $VIVADO_HLS_WORST_LATENCY"
printf "INFO: II (clk) : %-20s | %-20s\n" "Min: $VIVADO_HLS_MIN_II" "Max: $VIVADO_HLS_MAX_II"
printf "INFO: === Resources ===========================================================\n"
printf "INFO: BRAM : %-20s | %-20s | %-20s\n" "AVBL: $VIVADO_HLS_AVAILABLE_BRAM" "HLS: $VIVADO_HLS_BRAM ($VIVADO_HLS_BRAM_PERC%)" "LS: $VIVADO_BRAM ($VIVADO_BRAM_PERC%)"
printf "INFO: DSP  : %-20s | %-20s | %-20s\n" "AVBL: $VIVADO_HLS_AVAILABLE_DSP" "HLS: $VIVADO_HLS_DSP ($VIVADO_HLS_DSP_PERC%)" "LS: $VIVADO_DSP ($VIVADO_DSP_PERC%)"
printf "INFO: FF   : %-20s | %-20s | %-20s\n" "AVBL: $VIVADO_HLS_AVAILABLE_FF" "HLS: $VIVADO_HLS_FF ($VIVADO_HLS_FF_PERC%)" "LS: $VIVADO_FF ($VIVADO_FF_PERC%)"
printf "INFO: LUT  : %-20s | %-20s | %-20s\n" "AVBL: $VIVADO_HLS_AVAILABLE_LUT" "HLS: $VIVADO_HLS_LUT ($VIVADO_HLS_LUT_PERC%)" "LS: $VIVADO_LUT ($VIVADO_LUT_PERC%)"
printf "INFO: =========================================================================\n"

