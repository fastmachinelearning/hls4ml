#!/bin/bash

#
# Report the result of the HLS4ML design space exploration.
#

#
# The script inputs are:
# - input parameters (e.g. RF, model, report files, etc.)
# - Vivado HLS / Vivado report files (XML)
# - Vivado HLS / Vivado .log file
# - GNU parallel job-log file
#
# The script
# - uses an XML parser and awk to extract all of the DSE results
# - appends the results to a CSV file
#
# The script output is in following comma separated fortmat (CSV):
# - model
# - top module
# - architecture (or solution)
# - reuse factor
# - exit value
# - Vivado version
# - target FPGA part
# - Git revision
# - target clock period
# - estimated clock period
# - best latency
# - worst latency
# - minimum II
# - maximum II
# - Vivado HLS resource BRAM
# - Vivado HLS resource DSP
# - Vivado HLS resource FF
# - Vivado HLS resource LUT
# - Vivado resource BRAM_18K
# - Vivado resource DSP
# - Vivado resource FF
# - Vivado resource LUT
# - total execution time
# - Vivado HLS execution time
# - Vivado execution time
#

if [ ! $# -eq 7 ]; then
    echo "USAGE: $0 <jobs-log> <project-directory> <model> <reuse-factor> <max-execution-time> <CSV-file> <verbose>"
    exit 1
fi

JOB_LOG=$1

PROJECT_DIR=$2

MODEL=$3

REUSE_FACTOR=$4

TIMEOUT=$5

CSV_FILE=$6

VERBOSE=$7

if [ ! -f $JOB_LOG ]; then echo "ERROR: File $JOB_LOG does not exist!"; exit 1; fi

APP_FILE=$PROJECT_DIR/vivado_hls.app
if [ ! -f $APP_FILE ]; then echo "ERROR: File $APP_FILE does not exist!"; exit 1; fi

# If it does not exist, create a new CSV file and set the header line.
if [ ! -f $CSV_FILE ]; then
    echo -n "Model,ReuseFactor,VivadoHlsExitVal,VivadoExitVal,TotalExecutionTime,Timeout,TopModule,Arch," > $CSV_FILE
    echo -n "VivadoVersion,FpgaPart,GitRevision," >> $CSV_FILE
    echo -n "VivadoHlsTargetClk,VivadoHlsEstimatedClk,VivadoHlsBestLatency,VivadoHlsWorstLatency,VivadoHlsIIntervalMin,VivadoHlsIIntevalMax," >> $CSV_FILE
    echo -n "VivadoHlsResourceBram,VivadoHlsResourceDsp,VivadoHlsResourceFf,VivadoHlsResourceLut," >> $CSV_FILE
    echo -n "VivadoClk," >> $CSV_FILE
    echo -n "VivadoResourceBram,VivadoResourceDsp,VivadoResourceFf,VivadoResourceLut," >> $CSV_FILE
    echo -n "VivadoHlsExecutionTime,VivadoExecutionTime" >> $CSV_FILE
    echo "" >> $CSV_FILE
fi

#ARCHS=$(cat $APP_FILE | sed -e 's/ xmlns.*=".*"//g' | xmlstarlet sel -t -m "/project/solutions/solution" -v "@name" -n)
ARCHS="solution1"
for ARCH in $ARCHS; do
    if [ $VERBOSE ]; then
        echo "INFO: =========================================================================="
    fi
    VIVADO_HLS_REPORT_XML="$PROJECT_DIR/$ARCH/syn/report/csynth.xml"
    VIVADO_HLS_LOG="$PROJECT_DIR/../vivado_hls.log"
    VIVADO_REPORT_XML="$PROJECT_DIR/$ARCH/impl/report/verilog/myproject_export.xml"
    VIVADO_LOG="$PROJECT_DIR/$ARCH/impl/report/verilog/autoimpl.log"

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
    #TOTAL_EXECUTION_TIME=$(grep "Total elapsed time:" $PROJECT_DIR/../vivado_hls.log | awk '{print $7}')
    #if [ -z "$TOTAL_EXECUTION_TIME" ]; then EXECUTION_TIME=?; fi

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

    #
    # Check if the report files exist
    #
    [ -f "$VIVADO_REPORT_XML" ]
    VIVADO_EXIT_VAL=$?
    # as an alternative:  grep ERROR $VIVADO_LOG

    #
    # AWK on the job log from GNU parallel
    #
    VIVADO_HLS_EXIT_VAL=$(awk -v RF=$REUSE_FACTOR '$10 == RF { print $7 }' $JOB_LOG)
    if [ -z "$VIVADO_HLS_EXIT_VAL" ]; then VIVADO_HLS_EXIT_VAL=?; fi

    TOTAL_EXECUTION_TIME=$(awk -v RF=$REUSE_FACTOR '$10 == RF { print $4 }' $JOB_LOG)
    if [ -z "$TOTAL_EXECUTION_TIME" ]; then EXECUTION_TIME=?; fi

    if [ $VERBOSE ]; then
        echo "INFO: Project directory: $PROJECT_DIR"
        echo "INFO: Model: $MODEL"
        echo "INFO: Reuse Factor: $REUSE_FACTOR"
        echo "INFO: Vivado HLS Exit Value: $VIVADO_HLS_EXIT_VAL"
        echo "INFO: Vivado Exit Value: $VIVADO_EXIT_VAL"
        echo "INFO: Total Execution Time (secs): $TOTAL_EXECUTION_TIME"
        echo "INFO: Top module: $TOP_MODULE"
        echo "INFO: Architecture: $ARCH"

        echo "INFO: Vivado version: $VIVADO_VERSION"
        echo "INFO: FPGA part: $FPGA_PART"
        echo "INFO: Git revision: $GIT_REVISION"

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
    fi

    # Append the DSE results to CSV file.
    echo -n "$MODEL,$REUSE_FACTOR,$VIVADO_HLS_EXIT_VAL,$VIVADO_EXIT_VAL,$TOTAL_EXECUTION_TIME,$TIMEOUT,$TOP_MODULE,$ARCH," >> $CSV_FILE
    echo -n "$VIVADO_VERSION,$FPGA_PART,$GIT_REVISION," >> $CSV_FILE
    echo -n "$TARGET_CLK,$VIVADO_HLS_ESTIMATED_CLK,$VIVADO_HLS_BEST_LATENCY,$VIVADO_HLS_WORST_LATENCY,$VIVADO_HLS_IINTERVAL_MIN,$VIVADO_HLS_IINTERVAL_MAX," >> $CSV_FILE
    echo -n "$VIVADO_HLS_RESOURCE_BRAM,$VIVADO_HLS_RESOURCE_DSP,$VIVADO_HLS_RESOURCE_FF,$VIVADO_HLS_RESOURCE_LUT," >> $CSV_FILE
    echo -n "$VIVADO_ACHIEVED_CLK," >> $CSV_FILE
    echo -n "$VIVADO_RESOURCE_BRAM,$VIVADO_RESOURCE_DSP,$VIVADO_RESOURCE_FF,$VIVADO_RESOURCE_LUT," >> $CSV_FILE
    echo -n "$VIVADO_HLS_EXECUTION_TIME,$VIVADO_EXECUTION_TIME" >> $CSV_FILE
    echo "" >> $CSV_FILE
done

if [ $VERBOSE ]; then
    echo "INFO: =========================================================================="
fi
