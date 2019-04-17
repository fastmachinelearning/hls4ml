#!/bin/bash

if [ ! $# -eq 5 ]; then echo "Usage: $0 <project-directory> <model> <reuse-factor> <CSV-file>"; exit 1; fi

PROJECT_DIR=$1
APP_FILE=$PROJECT_DIR/vivado_hls.app
if [ ! -f $APP_FILE ]; then echo "File $APP_FILE does not exist!"; exit 1; fi

MODEL=$2

REUSE_FACTOR=$3

MAX_TIME=$4

CSV_FILE=$5
if [ ! -f $CSV_FILE ]; then echo "VivadoVersion,FpgaPart,TopModule,TargetClk,EstimatedClk,BestLatency,WorstLatency,IIntervalMin,IIntevalMax,HLSResourceBram,HLSResourceDsp,HLSResourceFf,HLSResourceLut,ResourceBram,ResourceDsp,ResourceFf,ResourceLut,GitRevision,Arch,Model,ReuseFactor,TotalExecutionTime,VivadoHlsExecutionTime,VivadoExecutionTime" > $CSV_FILE; fi

#ARCHS=$(cat $APP_FILE | sed -e 's/ xmlns.*=".*"//g' | xmlstarlet sel -t -m "/project/solutions/solution" -v "@name" -n)
ARCHS="solution1"

for ARCH in $ARCHS; do
    echo "==============================================================================="
    VIVADO_HLS_REPORT_XML="$PROJECT_DIR/$ARCH/syn/report/csynth.xml"
    VIVADO_REPORT_XML="$PROJECT_DIR/$ARCH/impl/report/verilog/myproject_export.xml"
    VIVADO_VERSION=$(xmllint --xpath "/profile/ReportVersion/Version/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_VERSION=?; fi
    FPGA_PART=$(xmllint --xpath "/profile/UserAssignments/Part/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then FPGA_PART=?; fi
    TOP_MODULE=$(xmllint --xpath "/profile/UserAssignments/TopModelName/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then TOP_MODULE=?; fi
    TARGET_CLK=$(xmllint --xpath "/profile/UserAssignments/TargetClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then TARGET_CLK=?; fi
    ESTIMATED_CLK=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then ESTIMATED_CLK=?; fi
    BEST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then BEST_LATENCY=?; fi
    WORST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then WORST_LATENCY=?; fi
    IINTERVAL_MIN=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-min/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then IINTERVAL_MIN=?; fi
    IINTERVAL_MAX=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-max/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then IINTERVAL_MAX=?; fi

    VIVADO_HLS_RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaEstimates/Resources/BRAM_18K/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_BRAM=?; fi
    VIVADO_HLS_RESOURCE_DSP=$(xmllint --xpath "/profile/AreaEstimates/Resources/DSP48E/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_DSP=?; fi
    VIVADO_HLS_RESOURCE_FF=$(xmllint --xpath "/profile/AreaEstimates/Resources/FF/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_FF=?; fi
    VIVADO_HLS_RESOURCE_LUT=$(xmllint --xpath "/profile/AreaEstimates/Resources/LUT/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_HLS_RESOURCE_LUT=?; fi

    ACHIEVED_CLK=$(xmllint --xpath "/profile/TimingReport/AchievedClockPeriod/text()" $VIVADO_HLS_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then ACHIEVED_CLK=?; fi

    VIVADO_RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaReport/Resources/BRAM/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_BRAM=?; fi
    VIVADO_RESOURCE_DSP=$(xmllint --xpath "/profile/AreaReport/Resources/DSP/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_DSP=?; fi
    VIVADO_RESOURCE_FF=$(xmllint --xpath "/profile/AreaReport/Resources/FF/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_FF=?; fi
    VIVADO_RESOURCE_LUT=$(xmllint --xpath "/profile/AreaReport/Resources/LUT/text()" $VIVADO_REPORT_XML 2> /dev/null)
    if [ ! $? -eq 0 ]; then VIVADO_RESOURCE_LUT=?; fi

    GIT_REVISION=$(git rev-parse --short HEAD)
    if [ ! $? -eq 0 ]; then echo "Git Error"; exit 1; fi

    TOTAL_EXECUTION_TIME=$(grep "Total elapsed time:" $PROJECT_DIR/../vivado_hls.log | awk '{print $7}')
    if [ -z "$TOTAL_EXECUTION_TIME" ]; then EXECUTION_TIME="?"; fi

    VIVADO_HLS_EXECUTION_TIME_STRING=$(grep "C/RTL SYNTHESIS COMPLETED IN" $PROJECT_DIR/../vivado_hls.log | awk '{print $7}')
    if [ -z "$VIVADO_HLS_EXECUTION_TIME_STRING" ]; then VIVADO_HLS_EXECUTION_TIME="?"; fi
    VIVADO_HLS_EXECUTION_TIME=$(echo $VIVADO_HLS_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')

    VIVADO_EXECUTION_TIME_STRING=$(grep "EXPORT IP COMPLETED IN" $PROJECT_DIR/../vivado_hls.log | awk '{print $6}')
    if [ -z "$VIVADO_EXECUTION_TIME_STRING" ]; then VIVADO_EXECUTION_TIME=?; fi
    VIVADO_EXECUTION_TIME=$(echo $VIVADO_EXECUTION_TIME_STRING | awk -F'[h|m|s]' '{ print ($1 * 3600) + ($2 * 60) + $3 }')

    echo "Vivado version: $VIVADO_VERSION"
    echo "FPGA part: $FPGA_PART"
    echo "Top module: $TOP_MODULE"
    echo "Target clock: $TARGET_CLK"
    echo "HLS Estimated clock: $ESTIMATED_CLK"
    echo "Achieved clock: $ESTIMATED_CLK"
    echo "Best Latency: $BEST_LATENCY"
    echo "Worst Latency: $WORST_LATENCY"
    echo "Min IInterval: $IINTERVAL_MIN"
    echo "Max IInterval: $IINTERVAL_MAX"
    echo "HLS BRAM: $VIVADO_HLS_RESOURCE_BRAM"
    echo "HLS DSP: $VIVADO_HLS_RESOURCE_DSP"
    echo "HLS FF: $VIVADO_HLS_RESOURCE_FF"
    echo "HLS LUT: $VIVADO_HLS_RESOURCE_LUT"
    echo "BRAM: $VIVADO_RESOURCE_BRAM"
    echo "DSP: $VIVADO_RESOURCE_DSP"
    echo "FF: $VIVADO_RESOURCE_FF"
    echo "LUT: $VIVADO_RESOURCE_LUT"
    echo "Git revision: $GIT_REVISION"
    echo "Architecture: $ARCH"
    echo "Model: $MODEL"
    echo "Reuse Factor: $REUSE_FACTOR"
    echo "Total Execution Time (secs): $TOTAL_EXECUTION_TIME"
    echo "Vivado HLS Execution Time (secs): $VIVADO_HLS_EXECUTION_TIME"
    echo "Vivado Execution Time (secs): $VIVADO_EXECUTION_TIME"

    echo "$VIVADO_VERSION,$FPGA_PART,$TOP_MODULE,$TARGET_CLK,$ESTIMATED_CLK,$BEST_LATENCY,$WORST_LATENCY,$IINTERVAL_MIN,$IINTERVAL_MAX,$VIVADO_HLS_RESOURCE_BRAM,$VIVADO_HLS_RESOURCE_DSP,$VIVADO_HLS_RESOURCE_FF,$VIVADO_HLS_RESOURCE_LUT,$VIVADO_RESOURCE_BRAM,$VIVADO_RESOURCE_DSP,$VIVADO_RESOURCE_FF,$VIVADO_RESOURCE_LUT,$GIT_REVISION,$ARCH,$MODEL,$REUSE_FACTOR,$TOTAL_EXECUTION_TIME,$VIVADO_HLS_EXECUTION_TIME,$VIVADO_EXECUTION_TIME" >> $CSV_FILE
done
echo "==============================================================================="
