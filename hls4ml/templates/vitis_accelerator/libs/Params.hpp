#pragma once

#include <ctype.h>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../kernel_wrapper.h"
#include "FpgaObj.hpp"

class Params {

  public:
    Params(int argc, char **argv) {
        int opt, temp;
        while ((opt = getopt(argc, argv, "x:vhr:n:i:o:d:c:b:")) != EOF)
            switch (opt) {
            case 'd':
                deviceBDFs.push_back(optarg);
                break;
            case 'x':
                xclbinFilename = optarg;
                break;
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            case 'c':
                temp = atoi(optarg);
                if (temp > 0 && temp < NUM_CU)
                    numCU = temp;
                break;
            case 'b':
                temp = atoi(optarg);
                if (temp > 0)
                    batchSize = temp;
                break;
            case 'n':
                temp = atoi(optarg);
                if (temp > 0)
                    numWorker = temp;
                break;
            case 'r':
                dataRepeatCount = atoi(optarg);
                break;
            case 'v':
                verbose++;
                break;
            case 'h':
                help();
                exit(0);
            default:
                std::cout << std::endl;
                abort();
            }

        if (verbose > 0)
            print();
    }

    void help(void) {
        std::cout << "Available options:" << std::endl;
        std::cout << "  -d <BDF>     : Specify device BDF (can be used multiple times)" << std::endl;
        std::cout << "  -x <path>    : Path to the XCLBIN file" << std::endl;
        std::cout << "  -b <size>    : Batch size (default = " << BATCHSIZE << ")" << std::endl;
        std::cout << "  -i <file>    : Input file path" << std::endl;
        std::cout << "  -o <file>    : Output file path" << std::endl;
        std::cout << "  -c <number>  : Maximum number of compute units to use" << std::endl;
        std::cout << "  -n <number>  : Maximum number of worker threads to use" << std::endl;
        std::cout << "  -r <number>  : Number of times to repeat input data" << std::endl;
        std::cout << "  -v           : Enable verbose output" << std::endl;
        std::cout << "  -h           : Display this help message" << std::endl;
    }

    void print(void) {
        std::cout << "Run parameters:" << std::endl;
        std::cout << "   xclbinFilename: " << xclbinFilename << std::endl;
        std::cout << "        batchSize: " << batchSize << std::endl;
        std::cout << "  sampleInputSize: " << sampleInputSize << std::endl;
        std::cout << " sampleOutputSize: " << sampleOutputSize << std::endl;
        std::cout << "            numCU: " << numCU << std::endl;
        std::cout << "    inputFilename: " << inputFilename << std::endl;
        std::cout << "   outputFilename: " << outputFilename << std::endl;
        std::cout << "        numWorker: " << numWorker << std::endl;
        std::cout << "  dataRepeatCount: " << dataRepeatCount << std::endl;
    }

    // Device
    std::vector<std::string> deviceBDFs;

    // Bitstream
    std::string xclbinFilename = "./build_hw_rel/kernel_wrapper.xclbin";
    size_t batchSize = BATCHSIZE;
    const size_t sampleInputSize = INSTREAMSIZE;
    const size_t sampleOutputSize = OUTSTREAMSIZE;
    size_t numCU = NUM_CU;

    // Data paths
    std::string inputFilename = "./tb_data/tb_input_features.dat";
    std::string referenceFilename = "tb_data/tb_output_predictions.dat";
    std::string outputFilename = "./tb_data/hw_results.dat";

    // Workers
    int numWorker = NUM_WORKER;

    // Benchmark
    int dataRepeatCount = -1;
    int verbose = 0;
};
