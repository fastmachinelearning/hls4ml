#include <string>

#include "FpgaObj.hpp"
#include "Params.hpp"
#include "Types.hpp"
#include "kernel_wrapper.h"
#include "xcl2.hpp"

void predict(double *input, uint64_t input_size, double *output, uint64_t output_size) {
    // TODO : Modify the databatcher so it can take those arrays instead of reading and writing files.

    return;
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
