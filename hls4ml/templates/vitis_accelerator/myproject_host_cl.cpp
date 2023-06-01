/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "xcl2.hpp" //host include

#include <algorithm> // host and test includes
#include <fstream>   // host and test includes
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

#include <sys/time.h>

//#include "kernel.h"
#define DATA_SIZE 2

#define CHECKPOINT 1

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

double get_time_diff(const struct timespec &start, const struct timespec &end) {
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
    return time_taken;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    cl_int err;
    cl::Context context;
    cl::Kernel myproject_kernel;
    cl::CommandQueue q;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr
    // is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create
    // its own host side buffer. So it is recommended to use this allocator if
    // user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR
    std::vector<int, aligned_allocator<int>> source_in1(DATA_SIZE);
    //    std::vector<int, aligned_allocator<int> > source_in2(DATA_SIZE);
    std::vector<int, aligned_allocator<int>> source_hw_results(DATA_SIZE);
    std::vector<int, aligned_allocator<int>> source_sw_results(DATA_SIZE);

    // Create the test data
    // std::generate(source_in1.begin(), source_in1.end(), std::rand);
    //    std::generate(source_in2.begin(), source_in2.end(), std::rand);
    for (int i = 0; i < DATA_SIZE; i++) {
        //        source_sw_results[i] = source_in1[i] + source_in2[i];
        source_hw_results[i] = 0;
    }

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, myproject_kernel = cl::Kernel(program, "myproject_kernel", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // [K] Code from hls4ml test
    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            // hls-fpga-machine-learning insert data
            // hls::stream<input_t> input_1("input_1");
            // nnet::copy_data<float, input_t, 0, N_INPUT_1_1>(in, input_1);
            // hls::stream<result_t> layer9_out("layer9_out");

            for (int i = 0; i < in.size(); i++) {
                source_in1.push_back(in[i]);
            }

            /*
            //[K] replace function call with kernel exection
                        // hls-fpga-machine-learning insert top-level-function
                        myproject_kernel(input_1,layer9_out);
            */
            // Allocate Buffer in Global Memory
            // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
            // Device-to-host communication
            OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 16 * vector_size_bytes,
                                                 source_in1.data(), &err));
            // OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
            //                                      source_in1.data(), &err));
            OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 5 * vector_size_bytes,
                                                    source_hw_results.data(), &err));

            uint32_t size = DATA_SIZE;
            OCL_CHECK(err, err = myproject_kernel.setArg(0, buffer_in1));
            // OCL_CHECK(err, err = myproject_kernel.setArg(1, buffer_in1));
            OCL_CHECK(err, err = myproject_kernel.setArg(1, buffer_output));
            OCL_CHECK(err, err = myproject_kernel.setArg(2, size));

            std::cout << "Migrate Objects to FPGA" << std::endl;
            // Copy input data to device global memory
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1}, 0)); // 0 means from host

            // Launch the Kernel
            // For HLS kernels global and local size is always (1,1,1). So, it is
            // recommended
            // to always use enqueueTask() for invoking HLS kernel
            std::cout << "Enqueue Task" << std::endl;
            OCL_CHECK(err, err = q.enqueueTask(myproject_kernel));

            // Copy Result from Device Global Memory to Host Local Memory
            std::cout << "Migrate Objects to host" << std::endl;
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
            std::cout << "Wait to finish" << std::endl;
            // OCL_CHECK(err, err = q.finish());
            q.finish();
            std::cout << "Finished" << std::endl;
            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for (int i = 0; i < N_LAYER_8; i++) {
                    std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                // nnet::print_result<result_t, N_LAYER_8>(layer9_out, std::cout, true);
                for (size_t i = 0; i < N_LAYER_8; i++) {
                    std::cout << source_hw_results[i] << " ";
                }
                std::cout << std::endl;
            }
            e++;
            std::cout << "e: " << e << std::endl;

            // hls-fpga-machine-learning insert tb-output
            // nnet::print_result<result_t, N_LAYER_8>(layer9_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
        /* [K] Do nothing when the input file can't be opened
                // hls-fpga-machine-learning insert zero
            hls::stream<input_t> input_1("input_1");
            nnet::fill_zero<input_t, N_INPUT_1_1>(input_1);
            hls::stream<result_t> layer9_out("layer9_out");

                // hls-fpga-machine-learning insert top-level-function
                //myproject_kernel(input_1,layer9_out);

                // hls-fpga-machine-learning insert output
                nnet::print_result<result_t, N_LAYER_8>(layer9_out, std::cout, true);

                // hls-fpga-machine-learning insert tb-output
                nnet::print_result<result_t, N_LAYER_8>(layer9_out, fout);
                std::cout<<"checkpoint"<<std::endl;
        */
    }

    fout.close();
    // std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;
    //[K] hls4ml code end

    //    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    //    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
    return 0;
}
