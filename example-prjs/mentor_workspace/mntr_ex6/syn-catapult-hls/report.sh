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

CSV_FILE=$PROJECT.$ARCH.csv

VERBOSE=2

REPORT_FILE=$PROJECT_DIR/rtl.rpt
if [ ! -f $REPORT_FILE ]; then echo "ERROR: File $REPORT_FILE does not exist!"; exit 1; fi

DESIGN_TOTAL_STRING=$(grep "Design Total" $REPORT_FILE)
CATAPULT_HLS_LATENCY=`echo $DESIGN_TOTAL_STRING | awk '{print $4}'`
CATAPULT_HLS_II=`echo $DESIGN_TOTAL_STRING | awk '{print $7}'`

TOTAL_AREA=$(grep "TOTAL AREA" $REPORT_FILE)

CATAPULT_HLS_DSP=`echo $TOTAL_AREA | awk '{print $6}'`
CATAPULT_HLS_LUT=`echo $TOTAL_AREA | awk '{print $7}'`
CATAPULT_HLS_MUX=`echo $TOTAL_AREA | awk '{print $8}'`

if [ $VERBOSE == 2 ]; then
    printf "INFO: === Info ================================================================\n"
#    printf "INFO: Project   : %-20s | %-20s | %-20s\n" "Dir: $PROJECT_DIR" "Top: $TOP_MODULE" "Arch: $ARCH"
#    printf "INFO: Vivado    : %-20s | %-20s\n" "Ver: $VIVADO_VERSION" "Part: $FPGA_PART"
#    printf "INFO: Git: $GIT_REVISION\n"
    printf "INFO: === Execution ===========================================================\n"
#    printf "INFO: Time (sec): Total: $TOTAL_EXECUTION_TIME\n"
#    printf "INFO: Time (sec): %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXECUTION_TIME" "LS: $VIVADO_EXECUTION_TIME" "RTL-sim: $RTL_SIMULATION_EXECUTION_TIME"
#    printf "INFO: Exit Value: %-20s | %-20s | %-20s\n" "HLS: $VIVADO_HLS_EXIT_VAL" "LS: $VIVADO_EXIT_VAL" "RTL-sim: $RTL_SIM_EXIT_VAL"
    printf "INFO: === Timing ==============================================================\n"
#    printf "INFO: Clock (ns)   : %-20s | %-20s | %-20s\n" "Target   : $TARGET_CLK" "HLS: $VIVADO_HLS_ESTIMATED_CLK" "LS: $VIVADO_ACHIEVED_CLK"
    printf "INFO: Latency (clk): %-20s\n" "$CATAPULT_HLS_LATENCY"
    printf "INFO: II (clk)     : %-20s\n" "$CATAPULT_HLS_II"
    printf "INFO: === Resources ===========================================================\n"
#    printf "INFO: BRAM : %-20s | %-20s | %-20s\n" "Available: $TARGET_RESOURCE_BRAM" "HLS: $VIVADO_HLS_RESOURCE_BRAM" "LS: $VIVADO_RESOURCE_BRAM"
    printf "INFO: DSP  : %-20s\n" "HLS: $CATAPULT_HLS_DSP"
    printf "INFO: LUT  : %-20s\n" "HLS: $CATAPULT_HLS_LUT"
    printf "INFO: MUX  : %-20s\n" "HLS: $CATAPULT_HLS_MUX"
    printf "INFO: =========================================================================\n"
fi
#
#
## Append the results to CSV file.
#echo -n "$VIVADO_HLS_EXIT_VAL,$RTL_SIM_EXIT_VAL,$VIVADO_EXIT_VAL,$TOTAL_EXECUTION_TIME,$TIMEOUT,$TOP_MODULE,$ARCH," >> $CSV_FILE
#echo -n "$VIVADO_VERSION,$FPGA_PART,$TARGET_RESOURCE_BRAM,$TARGET_RESOURCE_DSP,$TARGET_RESOURCE_FF,$TARGET_RESOURCE_LUT," >> $CSV_FILE
#echo -n "$TARGET_CLK,$VIVADO_HLS_ESTIMATED_CLK,$VIVADO_HLS_BEST_LATENCY,$VIVADO_HLS_WORST_LATENCY,$VIVADO_HLS_IINTERVAL_MIN,$VIVADO_HLS_IINTERVAL_MAX," >> $CSV_FILE
#echo -n "$VIVADO_HLS_RESOURCE_BRAM,$VIVADO_HLS_RESOURCE_DSP,$VIVADO_HLS_RESOURCE_FF,$VIVADO_HLS_RESOURCE_LUT," >> $CSV_FILE
#echo -n "$VIVADO_ACHIEVED_CLK," >> $CSV_FILE
#echo -n "$VIVADO_RESOURCE_BRAM,$VIVADO_RESOURCE_DSP,$VIVADO_RESOURCE_FF,$VIVADO_RESOURCE_LUT," >> $CSV_FILE
#echo -n "$VIVADO_HLS_EXECUTION_TIME,$VIVADO_EXECUTION_TIME,$RTL_SIMULATION_EXECUTION_TIME" >> $CSV_FILE
#echo "" >> $CSV_FILE
