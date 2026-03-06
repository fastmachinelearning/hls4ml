#!/bin/bash
set -e

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi

PROJECT=myproject
LIB_STAMP=mystamp

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"/src
BUILD_DIR="$(cd "$(dirname "$0")" && pwd)"/build
INC_FLAGS="-Isrc/hls/model_wrapper/firmware/ap_types/ -Isrc/hls/model_wrapper/"
WEIGHTS_DIR="\"${BASE_DIR}/hls/model_wrapper/firmware/weights\""

mkdir -p ${BUILD_DIR}
${CC} ${CFLAGS} ${INC_FLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${BASE_DIR}/hls/model_wrapper/firmware/${PROJECT}.cpp -o ${BUILD_DIR}/${PROJECT}.o
${CC} ${CFLAGS} ${INC_FLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${BASE_DIR}/${PROJECT}_bridge.cpp -o ${BUILD_DIR}/${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INC_FLAGS} -shared ${BUILD_DIR}/${PROJECT}.o ${BUILD_DIR}/${PROJECT}_bridge.o -o ${BUILD_DIR}/${PROJECT}-${LIB_STAMP}.so
rm -f ${BUILD_DIR}/*.o
