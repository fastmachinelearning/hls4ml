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
      
    std::cout << "Loading input data from tb_data/tb_input_features.dat" 
              << "and output predictions from tb_data/tb_output_features.dat" << std::endl;

    std::cout << "Writing output predictions to tb_data/tb_output_predictions.dat" << std::endl;
    
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    std::ifstream fin("tb_data/tb_input_features.dat");

    if (!fin.is_open()) {
        std::cerr << "Error: Could not open tb_input_features.dat" << std::endl;
    }

    if (!fpr.is_open()) {
        std::cerr << "Error: Could not open tb_output_predictions.dat" << std::endl;
    }

    std::vector<in_buffer_t> inputData;
    std::vector<out_buffer_t> outputPredictions;
    if (fin.is_open() && fpr.is_open()) {
        int e = 0;
        std::string iline;
        std::string pline;
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % 10 == 0) {
                std::cout << "Processing input/prediction " << e << std::endl;
            }
            std::stringstream in(iline); 
            std::stringstream pred(pline); 
            std::string token;
            while (in >> token) {
                in_buffer_t tmp = stof(token);
                inputData.push_back(tmp);
            }
            while (pred >> token) {
                out_buffer_t tmp = stof(token);
                outputPredictions.push_back(tmp);
            }
        }
        e++;
    }
    
    // Copying in testbench data
    int n = std::min((int) inputData.size(), INSTREAMSIZE * NUM_CU * NUM_THREAD);
    for (int i = 0; i < n; i++) {
        fpga.source_in[i] = inputData[i];
    }

    // Padding rest of buffer with arbitrary values
    for (int i = n; i < INSTREAMSIZE * NUM_CU * NUM_THREAD; i++) {
        fpga.source_in[i] = (in_buffer_t)(1234.567);
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
    
    std::cout << "Throughput = "
            << throughput
            <<" predictions/second\n" << std::endl;

    std::cout << "Writing hw resaults to file" << std::endl;
    std::ofstream resultsFile;
    resultsFile.open("tb_data/hw_results.dat", std::ios::trunc);
    if (resultsFile.is_open()) {   
        for (int i = 0; i < NUM_THREAD * NUM_CU * BATCHSIZE; i++) {
            std::stringstream line;
            for (int n = 0; n < DATA_SIZE_OUT; n++) {
                line << (float)fpga.source_hw_results[(i * DATA_SIZE_OUT) + n] << " ";
            }
            resultsFile << line.str() << "\n";
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