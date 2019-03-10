#!/bin/bash

#
# Stress test HLS4ML projects over the reuse factor.
#

# TODO: Can this script be ported/integrated in Jenkins?

# Model name.
#MODEL="KERAS_3layer"
MODEL="2layer_100x100"
# We assume the model files being
# KerasJson: ../example-keras-model-files/MODEL.json
# KerasH5:   ../example-keras-model-files/MODEL_weights.h5

# Begin, end and step for Reuse Factor.
RF_BEGIN=116
RF_END=200
RF_STEP=1

# Run at most $THREADS instances of Vivado.
THREADS=4

# Enable/disable Vivado HLS, Vivado (logic synthesis), and result collection.
RUN_HLS=1
RUN_LS=0
RUN_LOG=1

# Remove previous intermediate files.
RUN_CLEAN=1

# Let's use a working directory.
DIR=RF_stress_dir
mkdir -p $DIR

RESULT_FILE=RF_stress_results.csv

# Count how many tests.
let "test_count=0"

run_hls4ml_vivado ()
{
    test_count=$1
    rf=$2

    echo "Test # $test_count: ReuseFactor=$rf, Model:$MODEL"

    # Move to the working directory.
    cd $DIR
    if [ ! $? -eq 0 ]; then echo "Cannot find find directory $DIR"; continue; fi

    # Create HLS4ML configuration file (in the working directory).
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.yml
    fi
    sed "s/>>>REUSE<<</$rf/g" ../keras-config-REUSE-MODEL.yml | sed "s/>>>MODEL<<</$MODEL/g" > keras-config-$rf-$MODEL.yml
    if [ ! $? -eq 0 ]; then echo "Cannot create HLS4ML configuration file $DIR/keras-config-$rf-$MODEL.yml"; cd ..; continue; fi

    # Run HLS4ML generators.
    if [ $RUN_CLEAN -eq 1 ]; then
        rm -f keras-config-$rf-$MODEL.log
        rm -rf $MODEL\_RF$rf
    fi
    python ../keras-to-hls.py -c keras-config-$rf-$MODEL.yml > keras-config-$rf-$MODEL.log
    if [ ! $? -eq 0 ]; then echo "Cannot run HLS4ML generator on with the configuration file $DIR/keras-config-$rf-$MODEL.yml"; cd ..; continue; fi

    # Run Vivado HLS.
    if [ $RUN_HLS -eq 1 ]; then
        cd $MODEL\_RF$rf
        if [ ! $? -eq 0 ]; then echo "Cannot find find directory $MODEL\_RF$rf"; cd ../..; continue; fi
        # Kill Vivado HLS if does not return after 3 hours.
        timeout -k 30s 3h vivado_hls -f build_prj.tcl > /dev/null
        if [ ! $? -eq 0 ]; then echo "Vivado HLS failed in $DIR/$MODEL\_RF$rf"; cd ../..; continue; fi
        cd ..
    fi

#    # Run Vivado (it does not check if there was a previous HLS run).
#    if [ $RUN_LS -eq 1 ]; then
#        cd $MODEL\_RF$rf
#        if [ $RUN_CLEAN -eq 1 ]; then
#           rm -f run_vivado.tcl
#        fi
#        echo "open_project myproject_prj" > run_vivado.tcl
#        echo "export_design -flow syn -format ip_catalog" >> run_vivado.tcl
#        echo "exit" >> run_vivado.tcl
#        # Kill Vivado HLS if does not return after 3 hours.
#        timeout -k 30s 3h vivado_hls -l vivado.log -f run_vivado.tcl > /dev/null
#        if [ ! $? -eq 0 ]; then echo "Vivado failed in $DIR/$MODEL\_RF$rf"; cd ../..; continue; fi
#        cd ..
#    fi

     cd ..

}

# Iterate over the reuse factor value.
for rf_base in $(seq $(expr $RF_BEGIN) $THREADS $RF_END); do

    # Run parallel instances of Vivado.
    for rf in $(seq $rf_base $RF_STEP $(expr $rf_base + $THREADS - 1)); do
        let "test_count++"
        run_hls4ml_vivado $test_count $rf &
    done

    # Wait for all of the previous instances to terminate before collecting the
    # results and moving on to the next batch of runs.
    wait

    # Collect the results.
    for rf in $(seq $rf_base $RF_STEP $(expr $rf_base + $THREADS - 1)); do
        # Collect results (it does not check if there were HLS and LS runs).
        # TODO: Report script does not extract LS information.
        if [ $RUN_LOG -eq 1 ]; then
            ./parse-vivadohls-report.sh ./$DIR/$MODEL\_RF$rf/myproject_prj $MODEL $rf $RESULT_FILE
        fi
    done
done
