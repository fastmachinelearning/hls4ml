#!/bin/bash

#
# Explore HLS4ML design space over the reuse factor.
#

# TODO:
# - Can this script be ported/integrated in Jenkins?
# - Should we account for functional correctness?
# - Port to floating-point arithmetics the timing computation in the
#   information report
# - If logic synthesis is disabled in the configuration file, enable it

# Keep track of the current directory.
BASE_DIR="$PWD"

# ==============================================================================
# Model Configuration
# ==============================================================================

# Model directory.
MODEL_DIR="$BASE_DIR/../example-keras-model-files"

# Model name.
#MODEL="KERAS_3layer"
#MODEL="2layer_100x100"
#MODEL="KERAS_dense_16x100x100x100x100x100x5"
MODEL="KERAS_dense_16x200x200x200x200x200x5"
#MODEL="KERAS_dense_16x500x500x500x500x500x5"

# We assume the model files being:
# KerasJson: ../example-keras-model-files/MODEL.json
# KerasH5:   ../example-keras-model-files/MODEL_weights.h5

# Network characteristics.
N_IN=200
N_OUT=200

# ==============================================================================
# Directories and Files
# ==============================================================================

# Choose a partition where you have a lot of space. You can check it with:
# $ df -h
if [ -z "$SANDBOX_DIR" ]; then
    SANDBOX_DIR=$HOME/sandbox1
    #SANDBOX_DIR=$HOME/sandbox2
    #SANDBOX_DIR=$HOME/sandbox3
fi

# Use a working directory in a sandbox.
WORK_DIR="$SANDBOX_DIR/rf-exploration-dir/$MODEL"

# Use a separate directory for the reports.
RESULT_DIR="$BASE_DIR/reports"

# Output CSV file. See the "rf-reporting.sh" script for more details.
RESULT_FILE="$RESULT_DIR/$MODEL.csv"

# Output error file. See the "rf-error-reporting.sh" script for more details.
ERROR_FILE="$RESULT_DIR/$MODEL.errors.log"

# GNU parallel job log. See GNU parallel manual for more details.
# The logfile is in the following TAB separated format:
# - sequence number,
# - sshlogin,
# - start time as seconds since epoch,
# - run time in seconds,
# - bytes in files transferred,
# - bytes in files returned,
# - exit status,
# - signal,
# - command run
# We are particularly interested in the exit status.
JOB_LOG="$RESULT_DIR/$MODEL.jobs.log"

# ==============================================================================
# Design-Space-Exploration Configuration
# ==============================================================================

# Exploration mode.
# - Best-candidate mode [0]
#   use a formula to generate the best RF values given the network
#   architecture.
# - Brute-force mode    [1]
#   all of the reuse factors between RF_BEGIN and RF_END (with a RF_STEP) will
#   be tested. ATTENTION: Some values of reuse factor may cause very long
#   synthesis time.
# - User-defined mode   [2]
#   use a RF list provided by the user.
EXPLORATION_MODE=0

# Mode [1]
# Brute-force-mode configuration: begin, end and step for Reuse Factor.
RF_BEGIN=10
RF_END=100
RF_STEP=1

# Mode [2]
# User-defined RF values
USER_DEFINED_RF="100"

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
# 7h = 25200s
# 8h = 28800s
# 14h = 50400s
TIMEOUT_TIME=50400

# Run at most THREADS instances of Vivado HLS / Vivado.
THREADS=12

# ==============================================================================
# HLS, Logic Synthesis, Reports
# ==============================================================================

# Enable/disable Vivado HLS, Vivado (logic synthesis), and result collection.
RUN_HLS=1
RUN_LS=1
RUN_LOG=1

# Remove previous intermediate files.
RUN_CLEAN=1

# Enable [1] / disable [0] verbosity.
VERBOSE=0

# Enable [1] / disable [0] project compression (to save space).
COMPRESSION=0

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
    echo "INFO: Model: $MODEL"
    echo "INFO: Working directory: $WORK_DIR"
    echo "INFO: Result directory: $RESULT_DIR"
    echo "INFO: Candidate RFs: $candidates"
    echo "INFO: Candidate count: $candidate_count"
    echo "INFO: Job count: $THREADS (on a $cpus-thread CPU)"
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
setup_working_directories ()
{
    mkdir -p $WORK_DIR
    mkdir -p $RESULT_DIR
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
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot find find directory $WORK_DIR"; return 1; fi

    # Create HLS4ML configuration file (in the working directory).
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.yml
    fi
    sed "s#>>>REUSE<<<#$rf#g" $BASE_DIR/keras-config-REUSE-MODEL.yml | sed "s#>>>MODEL<<<#$MODEL#g" | sed "s#>>>MODEL_DIR<<<#$MODEL_DIR#g" > $WORK_DIR/keras-config-$rf-$MODEL.yml
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot create HLS4ML configuration file $WORK_DIR/keras-config-$rf-$MODEL.yml"; cd ..; return 1; fi

    # Run HLS4ML generators.
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.log
        rm -rf $MODEL\_RF$rf
    fi
    python $BASE_DIR/../keras-to-hls.py -c $WORK_DIR/keras-config-$rf-$MODEL.yml > $WORK_DIR/keras-config-$rf-$MODEL.log
    if [ ! $? -eq 0 ]; then echo "ERROR: Cannot run HLS4ML generator on with the configuration file $WORK_DIR/keras-config-$rf-$MODEL.yml"; cd ..; return; fi

    # Run Vivado HLS.
    if [ $RUN_HLS -eq 1 ]; then
        cd $MODEL\_RF$rf
        if [ ! $? -eq 0 ]; then echo "ERROR: Cannot find find directory $MODEL\_RF$rf"; cd ../..; return 1; fi
        #if [ $RUN_LS -eq 1 ]; then
        # TODO: enable logic synthesis (if disabled)
        #fi
        vivado_hls -f build_prj.tcl > /dev/null
        if [ ! $? -eq 0 ]; then echo "ERROR: Vivado HLS failed. See $WORK_DIR/$MODEL\_RF$rf/vivado_hls.log"; cd ../..; return 1; fi
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
        if [ $RUN_LOG -eq 1 ]; then
            bash rf-reporting.sh $JOB_LOG $WORK_DIR/$MODEL\_RF$rf/myproject_prj $MODEL $rf $TIMEOUT_TIME $RESULT_FILE $VERBOSE
        fi
    done
}

#
# Parse the Vivado HLS and Vivado (logic synthesis) reports and collect the
# error logs.
#
collect_errors ()
{
    # Collect the results.
    for rf in $(get_candidate_reuse_factors); do
        # Collect results (it does not check if there were HLS and LS runs).
        if [ $RUN_LOG -eq 1 ]; then
            bash rf-error-reporting.sh $JOB_LOG $WORK_DIR/$MODEL\_RF$rf/myproject_prj $MODEL $rf $ERROR_FILE $VERBOSE
        fi
    done
}



#
# Compress the Vivado HLS and Vivado project directories to save space.
#
compress_project_directories ()
{
    if [ $COMPRESSION == 1 ]; then
        # Collect the results.
        for rf in $(get_candidate_reuse_factors); do
            # Collect results (it does not check if there were HLS and LS runs).
            echo "INFO: compressing: $WORK_DIR/$MODEL\_RF$rf/myproject_prj"
            cd $WORK_DIR
            tar cvfz $MODEL\_RF$rf.tgz $MODEL\_RF$rf
            rm -rf $MODEL\_RF$rf
        done
    fi
}

# ==============================================================================
# The top of the hill :-)
# ==============================================================================

# These exports are necessary for GNU Parallel.
export MODEL
export RUN_LS
export RUN_HLS
export VERBOSE
export WORK_DIR
export BASE_DIR
export RUN_CLEAN
export MODEL_DIR
export RESULT_DIR
export RESULT_FILE
export ERROR_FILE
export TIMEOUT_TIME
export -f run_hls4ml_vivado

# Print some info, run the stress tests with GNU parallel, and collect the
# results.
print_info
setup_working_directories
get_candidate_reuse_factors | parallel --progress --will-cite --timeout $TIMEOUT_TIME --jobs $THREADS $SWAP --joblog $JOB_LOG run_hls4ml_vivado
collect_results
collect_errors
compress_project_directories
