#!/bin/bash

#
# Stress test HLS4ML projects over the reuse factor.
#

# TODO: Can this script be ported/integrated in Jenkins?

# Model name.
MODEL="KERAS_3layer"
# We assume the model files being
# KerasJson: ../example-keras-model-files/MODEL.json
# KerasH5:   ../example-keras-model-files/MODEL_weights.h5

# Begin, end and step for Reuse Factor.
RF_BEGIN=7
RF_END=64
RF_STEP=1

# Enable/disable Vivado HLS, Vivado (logic synthesis), and result collection.
RUN_HLS=1
RUN_LS=0
RUN_LOG=1

# Let's use a working directory.
DIR=stress-dir
mkdir -p $DIR

RESULT_FILE=stress_results.csv

# Count how many tests.
let "test_count=0"

# Iterate over reuse factor value.
for rf in $(seq $(expr $RF_BEGIN) $RF_STEP $RF_END); do

    # Keep trace of current test.
    let "test_count++"
    echo "Test # $test_count: ReuseFactor=$rf, Model:$MODEL"

    # Move to the working directory.
    cd $DIR
    if [ ! $? -eq 0 ]; then echo "Cannot find find directory $DIR"; continue; fi

    # Create HLS4ML configuration file (in the working directory).
    sed "s/>>>REUSE<<</$rf/g" ../keras-config-REUSE-MODEL.yml | sed "s/>>>MODEL<<</$MODEL/g" > keras-config-$rf-$MODEL.yml
    if [ ! $? -eq 0 ]; then echo "Cannot create HLS4ML configuration file $DIR/keras-config-$rf-$MODEL.yml"; cd ..; continue; fi

    # Run HLS4ML generators.
    python ../keras-to-hls.py -c keras-config-$rf-$MODEL.yml
    if [ ! $? -eq 0 ]; then echo "Cannot run HLS4ML generator on with the configuration file $DIR/keras-config-$rf-$MODEL.yml"; cd ..; continue; fi

    # Run Vivado HLS.
    if [ $RUN_HLS -eq 1 ]; then
        cd $MODEL\_RF$rf
        if [ ! $? -eq 0 ]; then echo "Cannot find find directory $MODEL\_RF$rf"; cd ../..; continue; fi
        vivado_hls -f build_prj.tcl
        if [ ! $? -eq 0 ]; then echo "Vivado HLS failed in $DIR/$MODEL\_RF$rf"; cd ../..; continue; fi
        cd ..
    fi

    # Run Vivado (it does not check if there was a previous HLS run).
    if [ $RUN_LS -eq 1 ]; then
        cd $MODEL\_RF$rf
        echo "open_project myproject_prj" > run_vivado.tcl
        echo "export_design -flow syn -format ip_catalog" >> run_vivado.tcl
        echo "exit" >> run_vivado.tcl
        vivado_hls -f run_vivado.tcl -l vivado.log
        if [ ! $? -eq 0 ]; then echo "Vivado failed in $DIR/$MODEL\_RF$rf"; cd ../..; continue; fi
        cd ..
    fi

    cd ..

    # Collect results (it does not check if there were HLS and LS runs).
    # TODO: Report script does not extract LS information.
    if [ $RUN_LOG -eq 1 ]; then
        ./parse-vivadohls-report.sh ./stress-dir/$MODEL\_RF$rf/myproject_prj $MODEL $rf $RESULT_FILE
    fi
done
