#!/bin/bash

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi
VITIS_UNIFIED_FLAGS="VITIS_UNIFIED"
CFLAGS="$CFLAGS -D$VITIS_UNIFIED_FLAGS"

INCFLAGS="-Ifirmware/ap_types/"

PROJECT=myprojectBaseName
WRAPPER_NAME=myprojectWrapName
LIB_STAMP=mystamp
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""

echo "------------- This is build_lib.sh debug message ----------------"
echo "Compiling for OSTYPE: $OSTYPE"
echo "CFLAGS: $CFLAGS"
echo "Include Flags: $INCFLAGS"
echo "Weights directory: $WEIGHTS_DIR"
echo "-----------------------------------------------------------------"

${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${WRAPPER_NAME}.cpp -o ${WRAPPER_NAME}.o
${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${WRAPPER_NAME}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so
rm -f *.o
