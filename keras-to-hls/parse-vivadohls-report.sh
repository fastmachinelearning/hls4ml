#!/bin/bash

if [ ! $# -eq 4 ]; then echo "Usage: $0 <project-directory> <model> <reuse-factor> <CSV-file>"; exit 1; fi

PROJECT_DIR=$1
APP_FILE=$PROJECT_DIR/vivado_hls.app
if [ ! -f $APP_FILE ]; then echo "File $APP_FILE does not exist!"; exit 1; fi

MODEL=$2

REUSE_FACTOR=$3

CSV_FILE=$4
if [ ! -f $CSV_FILE ]; then echo "VivadoVersion,FpgaPart,TopModule,TargetClk,EstimatedClk,BestLatency,WorstLatency,IIntervalMin,IIntevalMax,ResourceBram,ResourceDsp,ResourceFf,ResourceLut,GitRevision,Arch,Model,ReuseFactor,ExecutionTime" > $CSV_FILE; fi

#ARCHS=$(cat $APP_FILE | sed -e 's/ xmlns.*=".*"//g' | xmlstarlet sel -t -m "/project/solutions/solution" -v "@name" -n)
ARCHS="solution1"

for ARCH in $ARCHS; do
    echo "==============================================================================="
    XML_FILE="$PROJECT_DIR/$ARCH/syn/report/csynth.xml"
    VIVADO_VERSION=$(xmllint --xpath "/profile/ReportVersion/Version/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    FPGA_PART=$(xmllint --xpath "/profile/UserAssignments/Part/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    TOP_MODULE=$(xmllint --xpath "/profile/UserAssignments/TopModelName/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    TARGET_CLK=$(xmllint --xpath "/profile/UserAssignments/TargetClockPeriod/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    ESTIMATED_CLK=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    BEST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    WORST_LATENCY=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    IINTERVAL_MIN=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-min/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    IINTERVAL_MAX=$(xmllint --xpath "/profile/PerformanceEstimates/SummaryOfOverallLatency/Interval-max/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    RESOURCE_BRAM=$(xmllint --xpath "/profile/AreaEstimates/Resources/BRAM_18K/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    RESOURCE_DSP=$(xmllint --xpath "/profile/AreaEstimates/Resources/DSP48E/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    RESOURCE_FF=$(xmllint --xpath "/profile/AreaEstimates/Resources/FF/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    RESOURCE_LUT=$(xmllint --xpath "/profile/AreaEstimates/Resources/LUT/text()" $XML_FILE)
    if [ ! $? -eq 0 ]; then echo "Xmllint Error"; exit 1; fi
    GIT_REVISION=$(git rev-parse --short HEAD)
    if [ ! $? -eq 0 ]; then echo "Git Error"; exit 1; fi

    EXECUTION_TIME=$(grep "Total elapsed time:" $PROJECT_DIR/../vivado_hls.log | awk '{print $7}')
    if [ ! $? -eq 0 ]; then echo "Couldn't extract execution time"; exit 1; fi

    echo "Vivado version: $VIVADO_VERSION"
    echo "FPGA part: $FPGA_PART"
    echo "Top module: $TOP_MODULE"
    echo "Target clock: $TARGET_CLK"
    echo "Estimated clock: $ESTIMATED_CLK"
    echo "Best Latency: $BEST_LATENCY"
    echo "Worst Latency: $WORST_LATENCY"
    echo "Min IInterval: $IINTERVAL_MIN"
    echo "Max IInterval: $IINTERVAL_MAX"
    echo "BRAM: $RESOURCE_BRAM"
    echo "DSP: $RESOURCE_DSP"
    echo "FF: $RESOURCE_FF"
    echo "LUT: $RESOURCE_LUT"
    echo "Git revision: $GIT_REVISION"
    echo "Architecture: $ARCH"
    echo "Model: $MODEL"
    echo "Reuse Factor: $REUSE_FACTOR"
    echo "Execution Time (secs): $EXECUTION_TIME"

    echo "$VIVADO_VERSION,$FPGA_PART,$TOP_MODULE,$TARGET_CLK,$ESTIMATED_CLK,$BEST_LATENCY,$WORST_LATENCY,$IINTERVAL_MIN,$IINTERVAL_MAX,$RESOURCE_BRAM,$RESOURCE_DSP,$RESOURCE_FF,$RESOURCE_LUT,$GIT_REVISION,$ARCH,$MODEL,$REUSE_FACTOR,$EXECUTION_TIME" >> $CSV_FILE
done
echo "==============================================================================="
