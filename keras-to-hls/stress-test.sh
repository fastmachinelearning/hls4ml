#!/bin/bash

#
# This script stress tests HLS4ML projects when reuse factor changes.
#

# Let's pollute a sub-directory.
DIR=stress-dir
mkdir -p $DIR

# Model name.
MODEL="KERAS_3layer"
# We assume the model files being
# KerasJson: ../example-keras-model-files/MODEL.json
# KerasH5:   ../example-keras-model-files/MODEL_weights.h5

# Begin, end and step for Reuse Factor.
RF_BEGIN=16
RF_END=16
RF_STEP=1

RESULT_FILE=stress_result.csv

# Count how many tests.
let "test_count=0"

# Iterate over reuse factor value.
for rf in $(seq $(expr $RF_BEGIN - 1) $RF_STEP $RF_END); do
    let "test_count++" && \
        echo "Test # 1: ReuseFactor=$rf, Model:$MODEL" && \
        sed "s/>>>REUSE<<</$rf/g" keras-config-REUSE-MODEL.yml | sed "s/>>>MODEL<<</$MODEL/g" > $DIR/keras-config-$rf-$MODEL.yml && \
        cd $DIR && \
        python ../keras-to-hls.py -c keras-config-$rf-$MODEL.yml && \
        cd $MODEL\_RF$rf && \
        cp ../../Makefile.ini Makefile && \
        make && \
        cd ../.. && \
        ./parse-vivadohls-report.sh ./stress-dir/$MODEL\_RF$rf/myproject_prj $MODEL $rf $RESULT_FILE
done
