#!/bin/bash

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi
HLS_LIBS_PATH=mylibspath
LDFLAGS="-Wl,--no-undefined -Wl,--no-allow-shlib-undefined -Wl,--no-as-needed -Wl,-rpath,${HLS_LIBS_PATH}/lib/csim -L ${HLS_LIBS_PATH}/lib/csim -lhlsmc++-GCC46 -lhlsm-GCC46 -fno-builtin -fno-inline -Wl,-rpath,${HLS_LIBS_PATH}/tools/fpo_v7_0 -L ${HLS_LIBS_PATH}/tools/fpo_v7_0 -lgmp -lmpfr -lIp_floating_point_v7_0_bitacc_cmodel"
INCFLAGS="-Ifirmware/ap_types/"
PROJECT=myproject
LIB_STAMP=mystamp

${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so ${LDFLAGS}
rm -f *.o
