#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"

extern "C" {

// Wrapper of top level function for Python bridge
void myproject_float(
    //hls-fpga-machine-learning insert header #float
) {
    //hls-fpga-machine-learning insert wrapper #float
}

void myproject_double(
    //hls-fpga-machine-learning insert header #double
) {
    //hls-fpga-machine-learning insert wrapper #double
}

}

#endif
