#include <string>

#include "FpgaObj.hpp"
#include "Types.hpp"
#include "kernel_wrapper.h"
#include "xcl2.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                               << " <XCLBIN filename>"
                               << " [Profiling: Data repeat count]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string xclbinFilename = argv[1];
    int dataRepeatCount = -1;
    if (argc == 3) {
        dataRepeatCount = std::stoi(argv[2]);
    }

	FpgaObj</*INTERFACE_TYPES*/> fpga(BATCHSIZE,
                                            INSTREAMSIZE,
                                            OUTSTREAMSIZE,
                                            NUM_CU,
                                            xclbinFilename);

    fpga.createWorkers(NUM_WORKER,
                       FPGAType::/*FPGA_Type*/,
                       NUM_CHANNEL);

    if (dataRepeatCount == -1) {
        fpga.loadData("../tb_data/tb_input_features.dat");
    } else {
        fpga.loadData("../tb_data/tb_input_features.dat", true, dataRepeatCount);
    }

    fpga.evaluateAll();

    fpga.saveResults("../tb_data/hw_results.dat");

    return EXIT_SUCCESS;
}
