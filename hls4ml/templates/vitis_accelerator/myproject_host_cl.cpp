#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#include "kernel_wrapper.h"
#include "FpgaObj.hpp"
#include "HbmFpga.hpp"
#include "DdrFpga.hpp"
#include "timing.hpp"
#include "xcl2.hpp"

#define STRINGIFY(var) #var
#define EXPAND_STRING(var) STRINGIFY(var)


void runFPGAHelper(FpgaObj<in_buffer_t, out_buffer_t> &fpga) {
    std::stringstream ss;
    ss << (fpga.runFPGA()).str();
    fpga.write_ss_safe(ss.str());
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN Filename>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string xclbinFilename = argv[1];

    /*FPGATYPE*/<in_buffer_t, out_buffer_t> fpga(INSTREAMSIZE, OUTSTREAMSIZE, NUM_CU, NUM_THREAD, 100); 

    std::vector<cl::Device> devices = xcl::get_xil_devices();  // Utility API that finds xilinx platforms and return a list of devices connected to Xilinx platforms
    cl::Program::Binaries bins = xcl::import_binary_file(xclbinFilename);  // Load xclbin
    fpga.initializeOpenCL(devices, bins);

    fpga.allocateHostMemory(NUM_CHANNEL);
      
    std::cout << "Loading input data from tb_data/tb_input_features.dat" << std::endl;
    std::ifstream fin("tb_data/tb_input_features.dat");
    if (!fin.is_open()) {
        std::cerr << "Error: Could not open tb_input_features.dat" << std::endl;
    }
    std::vector<in_buffer_t> inputData;
    int num_inputs = 0;
    if (fin.is_open()) {
        std::string iline;
        while (std::getline(fin, iline)) {
            if (num_inputs % 10 == 0) {
                std::cout << "Processing input " << num_inputs << std::endl;
            }
            std::stringstream in(iline); 
            std::string token;
            while (in >> token) {
                in_buffer_t tmp = stof(token);
                inputData.push_back(tmp);
            }
            num_inputs++;
        }
    }
    
    // Copying in testbench data
    int num_samples = std::min(num_inputs, BATCHSIZE * NUM_CU * NUM_THREAD);
    memcpy(fpga.source_in.data(), inputData.data(), num_samples * DATA_SIZE_IN * sizeof(in_buffer_t));

    // Padding rest of buffer with arbitrary values
    for (int i = num_samples * DATA_SIZE_IN; i < INSTREAMSIZE * NUM_CU * NUM_THREAD; i++) {
        fpga.source_in[i] = (in_buffer_t)(2.345678);
    }

    std::vector<std::thread> hostAccelerationThreads;
    hostAccelerationThreads.reserve(NUM_THREAD);

    std::cout << "Beginning FPGA run" << std::endl;
    auto ts_start = SClock::now();

    for (int i = 0; i < NUM_THREAD; i++) {
        hostAccelerationThreads.push_back(std::thread(runFPGAHelper, std::ref(fpga)));
    }

    for (int i = 0; i < NUM_THREAD; i++) {
        hostAccelerationThreads[i].join();
    }

    fpga.finishRun();

    auto ts_end = SClock::now();
    float throughput = (float(NUM_CU * NUM_THREAD * 100 * BATCHSIZE) /
            float(std::chrono::duration_cast<std::chrono::nanoseconds>(ts_end - ts_start).count())) *
            1000000000.;
    std::cout << "Throughput = " << throughput <<" predictions/second\n" << std::endl;

    std::cout << "Writing hw results to file" << std::endl;
    std::ofstream resultsFile;
    resultsFile.open("tb_data/hw_results.dat", std::ios::trunc);
    if (resultsFile.is_open()) {   
        for (int i = 0; i < num_samples; i++) {
            std::stringstream oline;
            for (int n = 0; n < DATA_SIZE_OUT; n++) {
                oline << (float)fpga.source_hw_results[(i * DATA_SIZE_OUT) + n] << " ";
            }
            resultsFile << oline.str() << "\n";
        }
        resultsFile.close();
    } else {
        std::cerr << "Error writing hw results to file" << std::endl;
    }

    std::cout << "\nWriting run logs to file" << std::endl;
    std::ofstream outFile("u55c_executable_logfile.log", std::ios::trunc);
    if (outFile.is_open()) {
        outFile << fpga.ss.rdbuf();
        outFile.close();
    } else {
        std::cerr << "Error opening file for logging" << std::endl;
    }
    
    return EXIT_SUCCESS;
}