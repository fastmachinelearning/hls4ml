#!/bin/bash
set -e

CC=g++
CFLAGS="-O3 -fPIC -std=c++11"

# Include -fno-gnu-unique if it is there
if echo "" | ${CC} -Werror -fsyntax-only -fno-gnu-unique -xc++ - -o /dev/null &> /dev/null; then
  CFLAGS+=" -fno-gnu-unique"
fi

LDFLAGS=
INCFLAGS="-Ifirmware/ap_types/"
PROJECT=myproject
LIB_STAMP=mystamp
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""

${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so
rm -f *.o
