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

VERBOSE=2

APP_FILE=$PROJECT_DIR/vivado_hls.app
if [ ! -f $APP_FILE ]; then echo "ERROR: File $APP_FILE does not exist!"; exit 1; fi

echo -n "VivadoHlsExitVal,RTLSimExitVal,VivadoExitVal,TotalExecutionTime,Timeout,TopModule,Arch," > $CSV_FILE
echo -n "VivadoVersion,FpgaPart,TargetResourceBram,TargetResourceDsp,TargetResourceFf,TargetResourceLut," >> $CSV_FILE
echo -n "VivadoHlsTargetClk,VivadoHlsEstimatedClk,VivadoHlsBestLatency,VivadoHlsWorstLatency,VivadoHlsIIntervalMin,VivadoHlsIIntevalMax," >> $CSV_FILE
echo -n "VivadoHlsResourceBram,VivadoHlsResourceDsp,VivadoHlsResourceFf,VivadoHlsResourceLut," >> $CSV_FILE
echo -n "VivadoClk," >> $CSV_FILE
echo -n "VivadoResourceBram,VivadoResourceDsp,VivadoResourceFf,VivadoResourceLut," >> $CSV_FILE
echo -n "VivadoHlsExecutionTime,VivadoExecutionTime,RTLSimulationExecutionTime" >> $CSV_FILE
echo "" >> $CSV_FILE

ARCH="solution1"

VIVADO_HLS_REPORT_XML="$PROJECT_DIR/$ARCH/syn/report/csynth.xml"
VIVADO_HLS_LOG="$PROJECT_DIR/../vivado_hls.log"
VIVADO_REPORT_XML="$PROJECT_DIR/$ARCH/impl/report/verilog/$PROJECT"\_"export.xml"
VIVADO_LOG="$PROJECT_DIR/$ARCH/impl/report/verilog/autoimpl.log"
VIVADO_HLS_SOLUTION_LOG="$PROJECT_DIR/$ARCH/solution1.log"

#
# XML parser on Vivado HLS / Vivado reports
#
VIVADO_VERSION=$(xmllint --xpath "/profile/ReportVersion/Version/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_VERSION=?; fi

FPGA_PART=$(xmllint --xpath "/profile/UserAssignments/Part/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then FPGA_PART=?; fi

TOP_MODULE=$(xmllint --xpath "/profile/UserAssignments/TopModelName/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TOP_MODULE=?; fi

TARGET_CLK=$(xmllint --xpath "/profile/UserAssignments/TargetClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TARGET_CLK=?; fi

VIVADO_HLS_ESTIMATED_CLK=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_ESTIMATED_CLK=?; fi

VIVADO_HLS_BEST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_BEST_LATENCY=?; fi

VIVADO_HLS_WORST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_WORST_LATENCY=?; fi

VIVADO_HLS_IINTERVAL_MIN=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-min/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_IINTERVAL_MIN=?; fi

VIVADO_HLS_IINTERVAL_MAX=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-max/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_IINTERVAL_MAX=?; fi

VIVADO_HLS_RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaEstimates/Resources/BRAM_18K/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_BRAM=?; fi

VIVADO_HLS_RESOURCE_DSP=$(xmllint --xpath "/profile/AreaEstimates/Resources/DSP48E/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_DSP=?; fi

VIVADO_HLS_RESOURCE_FF=$(xmllint --xpath "/profile/AreaEstimates/Resources/FF/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_FF=?; fi

VIVADO_HLS_RESOURCE_LUT=$(xmllint --xpath "/profile/AreaEstimates/Resources/LUT/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_LUT=?; fi

TARGET_RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/BRAM_18K/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TARGET_RESOURCE_BRAM=?; fi

TARGET_RESOURCE_DSP=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/DSP48E/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TARGET_RESOURCE_DSP=?; fi

TARGET_RESOURCE_FF=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/FF/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TARGET_RESOURCE_FF=?; fi

TARGET_RESOURCE_LUT=$(xmllint --xpath "/profile/AreaEstimates/AvailableResources/LUT/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then TARGET_RESOURCE_LUT=?; fi

VIVADO_ACHIEVED_CLK=$(xmllint --xpath "/profile/TimingReport/AchievedClockPeriod/text()" $VIVADO_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_ACHIEVED_CLK=?; fi

VIVADO_RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaReport/Resources/BRAM/text()" $VIVADO_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_BRAM=?; fi

VIVADO_RESOURCE_DSP=$(xmllint --xpath "/profile/AreaReport/Resources/DSP/text()" $VIVADO_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_DSP=?; fi

VIVADO_RESOURCE_FF=$(xmllint --xpath "/profile/AreaReport/Resources/FF/text()" $VIVADO_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_FF=?; fi

VIVADO_RESOURCE_LUT=$(xmllint --xpath "/profile/AreaReport/Resources/LUT/text()" $VIVADO_REPORT_XML 2> /dev/null)
if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_LUT=?; fi

GIT_REVISION=$(git rev-parse --short HEAD)
if [ ! $? -eq 0 ]; then GIT_REVISION=?; fi

#
# AWK on Vivado HLS / Vivado logs
#
TOTAL_EXECUTION_TIME=$(grep "Total elapsed time:" $PROJECT_DIR/../vivado_hls.log | awk '{print $7}')
if [ -z "$TOTAL_EXECUTION_TIME" ]; then EXECUTION_TIME=?; fi

VIVADO_HLS_EXECUTION_TIME_STRING=$(grep "C/RTL SYNTHESIS COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
if [ -z "$VIVADO_HLS_EXECUTION_TIME_STRING" ]; then
    VIVADO_HLS_EXECUTION_TIME=?;
else
    VIVADO_HLS_EXECUTION_TIME=$(echo $VIVADO_HLS_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
fi

VIVADO_EXECUTION_TIME_STRING=$(grep "EXPORT IP COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
if [ -z "$VIVADO_EXECUTION_TIME_STRING" ]; then
    VIVADO_EXECUTION_TIME=?;
else
    VIVADO_EXECUTION_TIME=$(echo $VIVADO_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
fi

RTL_SIMULATION_EXECUTION_TIME_STRING=$(grep "C/RTL SIMULATION COMPLETED IN" $VIVADO_HLS_LOG | awk '{print $6}')
if [ -z "$RTL_SIMULATION_EXECUTION_TIME_STRING" ]; then
    RTL_SIMULATION_EXECUTION_TIME=?;
else
    RTL_SIMULATION_EXECUTION_TIME=$(echo $RTL_SIMULATION_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')
fi

VIVADO_HLS_EXIT_VAL=?
RTL_SIM_EXIT_VAL=?
VIVADO_EXIT_VAL=?

#
# Check HLS results [VIVADO_HLS_EXIT_VAL]
# - HLS Passed  [0]: XML file exists
# - HLS Failed  [1]: ERROR in Vivado HLS log file
# - HLS Timeout [2]: No XML and no ERROR in Vivado log file
#
if [ -f "$VIVADO_HLS_REPORT_XML" ]; then
    VIVADO_HLS_EXIT_VAL=0 # HLS Passed

    #
    # Check RTL simulation results [RTL_SIM_EXIT_VAL]
    # - Simulation Passed  [0]: 1 PASS in Vivado HLS log file.
    # - Simulation Failed  [1]: 0 PASS in Vivado HLS log file.
    #
    SIM_RESULTS=$(grep "C/RTL co-simulation finished: PASS" $VIVADO_HLS_LOG | wc -l)
    if [ $SIM_RESULTS == 1 ]; then # Simulation Passed
        RTL_SIM_EXIT_VAL=0

        #
        # Check Logic Synthesis results [VIVADO_EXIT_VAL]
        # - Logic Synthesis Passed  [0]: XML file exists
        # - Logic Synthesis Failed  [1]: No XML
        #
        if [ -f "$VIVADO_REPORT_XML" ]; then
            VIVADO_EXIT_VAL=0
        else # Logic Synthesis Failed or Timeout
            VIVADO_EXIT_VAL=1
        fi

    else # Simulation Failed
        SIM_RESULTS=$(grep "INFO: test FAIL" $VIVADO_HLS_LOG | wc -l)
        if [ $SIM_RESULTS -eq 1 ]; then # Failed
            RTL_SIM_EXIT_VAL=1
        else # Timeout
            RTL_SIM_EXIT_VAL=2
        fi
    fi

else # HLS Failed or Timeout
    HLS_RESULTS=$(grep "ERROR:" $VIVADO_HLS_LOG | wc -l)
    if [ $HLS_RESULTS -ge 1 ]; then # Failed
        VIVADO_HLS_EXIT_VAL=1
    else # Timeout
        VIVADO_HLS_EXIT_VAL=2
    fi
fi

if [ $VERBOSE == 1 ]; then
    echo "INFO: =========================================================================="
    echo "INFO: Project directory: $PROJECT_DIR"
#    echo "INFO: Reuse Factor: $REUSE_FACTOR"
    echo "INFO: Vivado HLS Exit Value: $VIVADO_HLS_EXIT_VAL"
    echo "INFO: RTL Simulation Exit Value: $RTL_SIM_EXIT_VAL"
    echo "INFO: Vivado Exit Value: $VIVADO_EXIT_VAL"
    echo "INFO: Total Execution Time (secs): $TOTAL_EXECUTION_TIME"
    echo "INFO: Top module: $TOP_MODULE"
    echo "INFO: Architecture: $ARCH"

    echo "INFO: Vivado version: $VIVADO_VERSION"
    echo "INFO: FPGA part: $FPGA_PART"
    echo "INFO: Git revision: $GIT_REVISION"
    echo "INFO: Target BRAM: $TARGET_RESOURCE_BRAM"
    echo "INFO: Target DSP: $TARGET_RESOURCE_DSP"
    echo "INFO: Target FF: $TARGET_RESOURCE_FF"
    echo "INFO: Target LUT: $TARGET_RESOURCE_LUT"

    echo "INFO: Target clock: $TARGET_CLK"
    echo "INFO: Vivado HLS Estimated clock: $VIVADO_HLS_ESTIMATED_CLK"
    echo "INFO: Vivado HLS Best Latency: $VIVADO_HLS_BEST_LATENCY"
    echo "INFO: Vivado HLS Worst Latency: $VIVADO_HLS_WORST_LATENCY"
    echo "INFO: Vivado HLS Min IInterval: $VIVADO_HLS_IINTERVAL_MIN"
    echo "INFO: Vivado HLS Max IInterval: $VIVADO_HLS_IINTERVAL_MAX"

    echo "INFO: Vivado HLS BRAM: $VIVADO_HLS_RESOURCE_BRAM"
    echo "INFO: Vivado HLS DSP: $VIVADO_HLS_RESOURCE_DSP"
    echo "INFO: Vivado HLS FF: $VIVADO_HLS_RESOURCE_FF"
    echo "INFO: Vivado HLS LUT: $VIVADO_HLS_RESOURCE_LUT"

    echo "INFO: Vivado Achieved clock: $VIVADO_ACHIEVED_CLK"

    echo "INFO: Vivado BRAM: $VIVADO_RESOURCE_BRAM"
    echo "INFO: Vivado DSP: $VIVADO_RESOURCE_DSP"
    echo "INFO: Vivado FF: $VIVADO_RESOURCE_FF"
    echo "INFO: Vivado LUT: $VIVADO_RESOURCE_LUT"

    echo "INFO: Vivado HLS Execution Time (secs): $VIVADO_HLS_EXECUTION_TIME"
    echo "INFO: Vivado Execution Time (secs): $VIVADO_EXECUTION_TIME"
    echo "INFO: RTL Simulation Execution Time (secs): $RTL_SIMULATION_EXECUTION_TIME"
    echo "INFO: =========================================================================="
fi

if [ $VERBOSE == 2 ]; then
    clear
    printf "INFO: === Info ================================================================\n"
    printf "INFO: Project   : %-20s | %-20s | %-20s\n" "Dir: $PROJECT_DIR" "Top: $TOP_MODULE" "Arch: $ARCH"
    printf "INFO: Vivado    : %-20s | %-20s\n" "Ver: $VIVADO_VERSION" "Part: $FPGA_PART"
    printf "INFO: Git: $GIT_REVISION\n"
    printf "INFO: === Execution ===========================================================\n"
    printf "INFO: Time (sec): Total: $TOTAL_EXECUTION_TIME\n"
    printf "INFO: Time (sec): %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXECUTION_TIME" "LS: $VIVADO_EXECUTION_TIME" "RTL-sim: $RTL_SIMULATION_EXECUTION_TIME"
    printf "INFO: Exit Value: %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXIT_VAL" "LS: $VIVADO_EXIT_VAL" "RTL-sim: $RTL_SIM_EXIT_VAL"
    printf "INFO: === Timing ==============================================================\n"
    printf "INFO: Clock (ns)   : %-20s | %-20s | %-20s\n" "Target   : $TARGET_CLK" "HLS: $VIVADO_HLS_ESTIMATED_CLK" "LS: $VIVADO_ACHIEVED_CLK"
    printf "INFO: Latency (clk): %-20s | %-20s\n" "Best: $VIVADO_HLS_BEST_LATENCY" "Worst: $VIVADO_HLS_WORST_LATENCY"
    printf "INFO: II (clk)     : %-20s | %-20s\n" "Min: $VIVADO_HLS_IINTERVAL_MIN" "Max: $VIVADO_HLS_IINTERVAL_MAX"
    printf "INFO: === Resources ===========================================================\n"
    printf "INFO: BRAM : %-20s | %-20s | %-20s\n" "Available: $TARGET_RESOURCE_BRAM" "HLS: $VIVADO_HLS_RESOURCE_BRAM" "LS: $VIVADO_RESOURCE_BRAM"
    printf "INFO: DSP  : %-20s | %-20s | %-20s\n" "Available: $TARGET_RESOURCE_DSP" "HLS: $VIVADO_HLS_RESOURCE_DSP" "LS: $VIVADO_RESOURCE_DSP"
    printf "INFO: FF   : %-20s | %-20s | %-20s\n" "Available: $TARGET_RESOURCE_FF" "HLS: $VIVADO_HLS_RESOURCE_FF" "LS: $VIVADO_RESOURCE_FF"
    printf "INFO: LUT  : %-20s | %-20s | %-20s\n" "Available: $TARGET_RESOURCE_LUT" "HLS: $VIVADO_HLS_RESOURCE_LUT" "LS: $VIVADO_RESOURCE_LUT"
    printf "INFO: =========================================================================\n"
fi


# Append the results to CSV file.
echo -n "$VIVADO_HLS_EXIT_VAL,$RTL_SIM_EXIT_VAL,$VIVADO_EXIT_VAL,$TOTAL_EXECUTION_TIME,$TIMEOUT,$TOP_MODULE,$ARCH," >> $CSV_FILE
echo -n "$VIVADO_VERSION,$FPGA_PART,$TARGET_RESOURCE_BRAM,$TARGET_RESOURCE_DSP,$TARGET_RESOURCE_FF,$TARGET_RESOURCE_LUT," >> $CSV_FILE
echo -n "$TARGET_CLK,$VIVADO_HLS_ESTIMATED_CLK,$VIVADO_HLS_BEST_LATENCY,$VIVADO_HLS_WORST_LATENCY,$VIVADO_HLS_IINTERVAL_MIN,$VIVADO_HLS_IINTERVAL_MAX," >> $CSV_FILE
echo -n "$VIVADO_HLS_RESOURCE_BRAM,$VIVADO_HLS_RESOURCE_DSP,$VIVADO_HLS_RESOURCE_FF,$VIVADO_HLS_RESOURCE_LUT," >> $CSV_FILE
echo -n "$VIVADO_ACHIEVED_CLK," >> $CSV_FILE
echo -n "$VIVADO_RESOURCE_BRAM,$VIVADO_RESOURCE_DSP,$VIVADO_RESOURCE_FF,$VIVADO_RESOURCE_LUT," >> $CSV_FILE
echo -n "$VIVADO_HLS_EXECUTION_TIME,$VIVADO_EXECUTION_TIME,$RTL_SIMULATION_EXECUTION_TIME" >> $CSV_FILE
echo "" >> $CSV_FILE
