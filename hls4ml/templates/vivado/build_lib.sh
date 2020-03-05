#!/bin/bash

CC=g++
CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
LDFLAGS=
INCFLAGS="-Ifirmware/ap_types/"
PROJECT=myproject

${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}.so
rm -f *.o
