#include <string>
#include <cstdio>
#include <iostream>

#include "FpgaObj.hpp"
#include "Params.hpp"
#include "Types.hpp"
#include "kernel_wrapper.h"
#include "xcl2.hpp"

extern "C" void predict(double *input, uint64_t input_size, double *output, uint64_t output_size) {

    int argc = 2;
    char *argv[] = {const_cast<char *>("host"), const_cast<char *>("myproject.xclbin")};

    // Redirect stdout to a file - For debugging purposes
    freopen("c_log.txt","w",stdout);
    std::cout << "Entered shared function !" << std::endl;

    Params params(argc, argv);

    FpgaObj<in_buffer_t, out_buffer_t> fpga(params);

    fpga.createWorkers(params.numWorker);

    fpga.loadSharedData(input, input_size);

    fpga.evaluateAll();

    fpga.checkResults(params.referenceFilename);

    fpga.returnSharedResults(output, output_size);
}

int main(int argc, char **argv) {

    Params params(argc, argv);

    FpgaObj</*INTERFACE_TYPES*/> fpga(params);

    fpga.createWorkers(params.numWorker);

    fpga.loadData(params.inputFilename, params.dataRepeatCount);

    fpga.evaluateAll();

    fpga.checkResults(params.referenceFilename);

    fpga.saveResults(params.outputFilename);

    return EXIT_SUCCESS;
}
