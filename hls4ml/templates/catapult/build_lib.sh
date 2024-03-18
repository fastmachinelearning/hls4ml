#!/bin/bash

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "linux"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique -Wno-pragmas"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi
LDFLAGS=

# Pick up AC libraries from Catapult install first
INCFLAGS="-I$MGC_HOME/shared/include -I$MGC_HOME/shared/include/nnet_utils -Ifirmware/ac_types/include -Ifirmware/ac_math/include -Ifirmware/ac_simutils/include -Ifirmware/nnet_utils"
PROJECT=myproject
LIB_STAMP=mystamp

${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so
rm -f *.o
