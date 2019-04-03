#!/bin/bash

#
# Stress test HLS4ML projects over the reuse factor.
#

# TODO:
# - Can this script be ported/integrated in Jenkins?
# - Catch error code from GNU parallel and add that to CSV report (it can be
#   useful to distinguish timeouts from Vivado crashes)
# - Port to floating-point arithmetics the timing computation in the
#   information report
# - If logic synthesis is disabled in the configuration file, enable it

# ==============================================================================
# Model Configuration
# ==============================================================================

# Model name.
#MODEL="KERAS_3layer"
#MODEL="2layer_100x100"
MODEL="KERAS_dense_16x100x100x100x100x100x5"
#MODEL="KERAS_dense_16x200x200x200x200x200x5"
#MODEL="KERAS_dense_16x500x500x500x500x500x5"

# We assume the model files being:
# KerasJson: ../example-keras-model-files/MODEL.json
# KerasH5:   ../example-keras-model-files/MODEL_weights.h5

# Network characteristics.
N_IN=100
N_OUT=100

# ==============================================================================
# Design-Space-Exploration Configuration
# ==============================================================================

# Exploration mode.
# - Best-candidate mode [0]
#   use a formula to generate the best values for RF given the network
#   architecture.
# - Brute-force mode    [1]
#   all of the reuse factors between RF_BEGIN and RF_END (with a RF_STEP) will
#   be tested. ATTENTION: Some values of reuse factor may cause very long
#   synthesis time.
# - User-defined mode   [2]
#   use a RF list provided by the user.
EXPLORATION_MODE=0

# Brute-force-mode configuration: begin, end and step for Reuse Factor.
RF_BEGIN=10
RF_END=100
RF_STEP=1

# User-defined RF values
USER_DEFINED_RF="1 2 4 5 8 16 40 80 125 200 250"

# ==============================================================================
# Host constraints
# ==============================================================================

# Max execution time.
# 1h = 3600s
# 2h = 7200s
# 3h = 10800s
# 4h = 14400s
# 5h = 18000s
# 6h = 21600s
TIMEOUT_TIME=21600

# Run at most THREADS instances of Vivado HLS / Vivado.
THREADS=6

# ==============================================================================
# HLS, Logic Synthesis, Reports
# ==============================================================================

# Enable/disable Vivado HLS, Vivado (logic synthesis), and result collection.
RUN_HLS=1
RUN_LS=1
RUN_LOG=1

# Remove previous intermediate files.
RUN_CLEAN=1

# ==============================================================================
# Files and directories
# ==============================================================================

# Let's use a working directory.
WORK_DIR=RF_stress_dir_$MODEL

# Output CSV file.
RESULT_FILE=RF_stress_results_$MODEL.csv

# ==============================================================================
#
# GNU Parallel configuration.
#
# This iteration of the "RF stress script" uses GNU Parallel.
#
# See 'man parallel' for details
#
# This is the first time I found a licensing disclaimer like this:
#
# Academic tradition requires you to cite works you base your article on.
# When using programs that use GNU Parallel to process data for publication
# please cite:
#
#  O. Tange (2011): GNU Parallel - The Command-Line Power Tool,
#    ;login: The USENIX Magazine, February 2011:42-47.
#
#    This helps funding further development; AND IT WON'T COST YOU A CENT.
#    If you pay 10000 EUR you should feel free to use GNU Parallel without citing.
#
# ==============================================================================

# Do not swap.
#SWAP=--noswap

# ==============================================================================
# Functions
# ==============================================================================

#
# Print some general information on the console.
#
print_info ()
{
    if [ $EXPLORATION_MODE == 0 ]; then # best-candidate mode
        echo "INFO: ==============================================================================="
        echo "INFO: Best-Candidate Mode"
        echo "INFO: ==============================================================================="
        echo "INFO: Network dimensions: N_IN=$N_IN, N_OUT=$N_OUT"
    elif [ $EXPLORATION_MODE == 1 ]; then # brute-force mode
        echo "INFO: ==============================================================================="
        echo "INFO: Brute-Force Mode"
        echo "INFO: ==============================================================================="
        echo "INFO: RF: begin $RF_BEGIN, end $RF_END, step $RF_STEP"
    else # user-defined mode
        echo "INFO: ==============================================================================="
        echo "INFO: User-Defined Mode"
        echo "INFO: ==============================================================================="
    fi
    candidates=$(get_candidate_reuse_factors | tr '\n' ' ')
    candidate_count=$(echo $candidates | wc -w)
    cpus=$(lscpu | grep -E '^CPU\(' | awk '{print $2}')
    memory_kb=$(vmstat -s | grep "total memory" | awk '{print $1}')
    memory_mb=$((memory_kb / 1024))
    # TODO: port to floating-point arithmetics
    MAX_TIME_HH=$(((TIMEOUT_TIME / 60) / 60))
    MAX_TIME_MM=$(((TIMEOUT_TIME % 60) / 60))
    MAX_TIME_SS=$(((TIMEOUT_TIME % 60) % 60))
    OVERALL_MAX_TIME=$(((TIMEOUT_TIME * candidate_count) / THREADS))
    OVERALL_MAX_TIME_HH=$(((OVERALL_MAX_TIME / 60) / 60))
    OVERALL_MAX_TIME_MM=$(((OVERALL_MAX_TIME % 60) / 60))
    OVERALL_MAX_TIME_SS=$(((OVERALL_MAX_TIME % 60) % 60))
    echo "INFO: Candidate RFs: $candidates"
    echo "INFO: Candidate # $candidate_count"
    echo "INFO: Job # $THREADS on a $cpus-thread CPU"
    echo "INFO: Maximum available memory: $memory_kb KB (= $memory_mb MB)"
    echo "INFO: Single-run timeout*: $TIMEOUT_TIME secs (= $MAX_TIME_HH hours, $MAX_TIME_MM mins, $MAX_TIME_SS secs) * Worst case"
    echo "INFO: Total maximum time*: $OVERALL_MAX_TIME secs (= $OVERALL_MAX_TIME_HH hours, $OVERALL_MAX_TIME_MM mins, $OVERALL_MAX_TIME_SS secs) * Worst case"
    echo -n "INFO: Do you want to proceed? [yes/NO]: "
    read answer
    if [ ! "$answer" == "yes" ]; then
        echo "INFO: Terminated by the user"
        exit 0
    fi
}

#
# Print the candidate reuse factors on the output console.
#
# If brute-force mode is enabled, it prints all of the values between RF_BEGIN and
# RF_END with a RF_STEP. The total number of values are ((RF_END - RF_BEGIN) /
# RF_STEP).
#
# If best-candidate mode is enabled, it prints all of the 'rf' values that
# satisfy the equation (((N_IN * N_OUT) % rf) == 0).
#
get_candidate_reuse_factors ()
{
    if [ $EXPLORATION_MODE == 0 ]; then # best-candidate mode
        for i in $(seq 1 $((N_IN * N_OUT))); do if [ $(((N_IN * N_OUT) % $i)) == 0 ]; then echo $i; fi; done
    elif [ $EXPLORATION_MODE == 1 ]; then # brute-force mode
        seq $RF_BEGIN $RF_STEP $RF_END
    else # user-defined mode
        for i in $USER_DEFINED_RF; do echo $i; done
    fi
}

#
# Setup working directory
#
setup_working_directory ()
{
    mkdir -p $WORK_DIR
}

#
# Run Vivado HLS and Vivado (logic synthesis).
#
run_hls4ml_vivado ()
{
    rf=$1

    echo "INFO: Stress ReuseFactor=$rf, Model:$MODEL"

    # Move to the working directory.
    cd $WORK_DIR
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot find find directory $WORK_DIR"; return; fi

    # Create HLS4ML configuration file (in the working directory).
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.yml
    fi
    sed "s/>>>REUSE<<</$rf/g" ../keras-config-REUSE-MODEL.yml | sed "s/>>>MODEL<<</$MODEL/g" > keras-config-$rf-$MODEL.yml
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot create HLS4ML configuration file $WORK_DIR/keras-config-$rf-$MODEL.yml"; cd ..; return; fi

    # Run HLS4ML generators.
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.log
        rm -rf $MODEL\_RF$rf
    fi
    python ../keras-to-hls.py -c keras-config-$rf-$MODEL.yml > keras-config-$rf-$MODEL.log
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot run HLS4ML generator on with the configuration file $WORK_DIR/keras-config-$rf-$MODEL.yml"; cd ..; return; fi

    # Run Vivado HLS.
    if [ $RUN_HLS -eq 1 ]; then
        cd $MODEL\_RF$rf
        if [ ! $? -eq 0 ]; then echo "ERROR: Cannot find find directory $MODEL\_RF$rf"; cd ../..; return; fi
        #if [ $RUN_LS -eq 1 ]; then
        # TODO: enable logic synthesis (if disabled)
        #fi
        # Kill Vivado HLS if does not return after 3 hours.
        vivado_hls -f build_prj.tcl > /dev/null
        if [ ! $? -eq 0 ]; then echo "ERROR: Vivado HLS failed. See $WORK_DIR/$MODEL\_RF$rf/vivado_hls.log"; cd ../..; return; fi
        cd ..
    fi

    cd ..

}

#
# Parse the Vivado HLS and Vivado (logic synthesis) reports and collect the
# results in a CSV file.
#
collect_results ()
{
    # Collect the results.
    for rf in $(get_candidate_reuse_factors); do
        # Collect results (it does not check if there were HLS and LS runs).
        # TODO: Report script does not extract LS information.
        if [ $RUN_LOG -eq 1 ]; then
            ./parse-vivadohls-report.sh ./$WORK_DIR/$MODEL\_RF$rf/myproject_prj $MODEL $rf $TIMEOUT_TIME $RESULT_FILE
        fi
    done
}

# ==============================================================================
# The top of the hill :-)
# ==============================================================================

# These exports are necessary for GNU Parallel.
export -f run_hls4ml_vivado
export MODEL
export WORK_DIR
export RUN_CLEAN
export RUN_HLS
export RUN_LS
export TIMEOUT_TIME

# Print some info, run the stress tests with GNU parallel, and collect the
# results.
print_info
setup_working_directory
get_candidate_reuse_factors | parallel --progress --will-cite --timeout $TIMEOUT_TIME --jobs $THREADS $SWAP run_hls4ml_vivado
collect_results
