#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "DdrFpga.hpp"
#include "FpgaObj.hpp"
#include "HbmFpga.hpp"
#include "kernel_wrapper.h"
#include "xcl2.hpp"

#define STRINGIFY(var) #var
#define EXPAND_STRING(var) STRINGIFY(var)

void runFPGAHelper(FpgaObj</*IN_INTERFACE_TYPE*/, /*OUT_INTERFACE_TYPE*/> &fpga) { fpga.runFPGA(); }

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN Filename>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string xclbinFilename = argv[1];

    // hls-fpga-machine-learning FPGA type

    std::vector<cl::Device> devices = xcl::get_xil_devices(); // Utility API that finds xilinx platforms and return a list of
                                                              // devices connected to Xilinx platforms
    auto fileBuf = xcl::read_binary_file(xclbinFilename);     // Load xclbin
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    fpga.initializeOpenCL(devices, bins);

    fpga.allocateHostMemory(NUM_CHANNEL);

    std::cout << "Loading input data from tb_data/tb_input_features.dat" << std::endl;
    std::ifstream fin("tb_data/tb_input_features.dat");
    if (!fin.is_open()) {
        std::cerr << "Error: Could not open tb_input_features.dat" << std::endl;
    }
    std::vector</*IN_INTERFACE_TYPE*/> inputData;
    int num_inputs = 0;
    if (fin.is_open()) {
        std::string iline;
        while (std::getline(fin, iline)) {
            if (num_inputs % 100 == 0) {
                std::cout << "Processing input " << num_inputs << std::endl;
            }
            std::stringstream in(iline);
            std::string token;
            while (in >> token) {
                inputData.push_back(/*IN_TYPE_CAST*/ stof(token));
            }
            num_inputs++;
        }
    }
    fin.close();

    // Copying input data into memory-mapped arrays
    int num_samples = std::min(num_inputs, BATCHSIZE * NUM_CU * NUM_THREAD);
    memcpy(fpga.source_in.data(), inputData.data(), num_samples * INSTREAMSIZE * sizeof(/*IN_INTERFACE_TYPE*/));

    std::vector<std::thread> hostAccelerationThreads;
    hostAccelerationThreads.reserve(NUM_THREAD);

    std::cout << "Beginning FPGA run" << std::endl;
    auto ts_start = std::chrono::system_clock::now();

    for (int i = 0; i < NUM_THREAD; i++) {
        hostAccelerationThreads.push_back(std::thread(runFPGAHelper, std::ref(fpga)));
    }

    for (int i = 0; i < NUM_THREAD; i++) {
        hostAccelerationThreads[i].join();
    }

    fpga.finishRun();

    auto ts_end = std::chrono::system_clock::now();
    float throughput = (float(BATCHSIZE * NUM_CU * NUM_THREAD * 10) /
                        float(std::chrono::duration_cast<std::chrono::nanoseconds>(ts_end - ts_start).count())) *
                       1000000000.;
    std::cout << "Throughput = " << throughput << " predictions/second\n" << std::endl;

    std::cout << "Writing hw results to file" << std::endl;
    std::ofstream resultsFile;
    resultsFile.open("tb_data/hw_results.dat", std::ios::trunc);
    if (resultsFile.is_open()) {
        for (int i = 0; i < num_samples; i++) {
            std::stringstream oline;
            for (int n = 0; n < DATA_SIZE_OUT; n++) {
                oline << /*OUT_TYPE_CAST*/ fpga.source_hw_results[(i * DATA_SIZE_OUT) + n] << " ";
            }
            resultsFile << oline.str() << "\n";
        }
        resultsFile.close();
    } else {
        std::cerr << "Error writing hw results to file" << std::endl;
    }

    return EXIT_SUCCESS;
}
