#include "myproject.h"
#include "parameters.h"

// hls-fpga-machine-learning insert weights

void MyProject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    auto inputsArr = InPipe::read();

// hls-fpga-machine-learning insert layers

// hls-fpga-machine-learning return

    OutPipe::write(outData);
}


