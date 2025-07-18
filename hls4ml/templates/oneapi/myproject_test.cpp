#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_data_movement.h"
#include "firmware/parameters.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

#if (__INTEL_CLANG_COMPILER < 20250000)
#include <sycl/ext/intel/prototype/interfaces.hpp>
#endif

#include "exception_handler.hpp"
// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

#if not defined(IS_BSP)
using sycl::ext::intel::experimental::property::usm::buffer_location;
#endif

// Functions that reads input and prediction data from files.
// Returns `true` if files are read successfully and not empty.
// Returns `false` otherwise.
bool prepare_data_from_file(std::string &fin_path, std::string &fpr_path, std::vector<std::vector<float>> &inputs,
                            std::vector<std::vector<float>> &predictions) {
    // load input data from text file
    std::ifstream fin(fin_path.c_str());
    // load predictions from text file
    std::ifstream fpr(fpr_path.c_str());

    std::string iline;
    std::string pline;

    if (fin.is_open() && fpr.is_open()) {
        size_t num_iterations = 0;

        // Prepare input data from file. Load predictions from file.
        for (; std::getline(fin, iline) && std::getline(fpr, pline); num_iterations++) {
            if (num_iterations % CHECKPOINT == 0) {
                std::cout << "Processing input " << num_iterations << std::endl;
            }

            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::stringstream ssin(iline);
            while (ssin >> current) {
                in.push_back(current);
            }

            std::stringstream sspred(pline);
            while (sspred >> current) {
                pr.push_back(current);
            }

            std::copy(pr.cbegin(), pr.cend(), predictions.back().begin());
            std::copy(in.cbegin(), in.cend(), inputs.back().begin());
        }
        fin.close();
        fpr.close();
        if (inputs.empty())
            return false;
        else
            return true;
    } else {
        return false;
    }
}

int main(int argc, char **argv) {

#if FPGA_SIMULATOR
#define NUM_ITERATIONS 5
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
#define NUM_ITERATIONS 100
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
#define NUM_ITERATIONS 10
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    // make sure the device supports USM host allocations
    if (!device.has(sycl::aspect::usm_host_allocations)) {
        std::cerr << "This design must either target a board that supports USM "
                     "Host/Shared allocations, or IP Component Authoring. "
                  << std::endl;
        std::terminate();
    }

    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

    std::string INPUT_FILE = "tb_data/tb_input_features.dat";
    std::string PRED_FILE = "tb_data/tb_output_predictions.dat";
    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    // Allocate vectors on stack to hold data from files temporarily.
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> predictions;
    bool file_valid = prepare_data_from_file(INPUT_FILE, PRED_FILE, inputs, predictions);
    unsigned int num_iterations;
    if (file_valid) {
        num_iterations = inputs.size();
    } else {
        num_iterations = NUM_ITERATIONS;
    }

    // hls-fpga-machine-learning insert runtime contant

    try {
#if defined(IS_BSP)
        // Allocate host memory if BSP is in use.
        float *vals = sycl::malloc_host<float>(kInputSz, q);
        if (vals == nullptr) {
            std::cerr << "ERROR: host allocation failed for input\n";
            fout.close();
            return 1;
        }
        float *outputs = sycl::malloc_host<float>(kOutputSz, q);
        if (outputs == nullptr) {
            std::cerr << "ERROR: host allocation failed for output\n";
            fout.close();
            return 1;
        }
#else
        float *vals =
            sycl::malloc_shared<float>(kInputSz, q, sycl::property_list{buffer_location(nnet::kInputBufferLocation)});
        float *outputs =
            sycl::malloc_shared<float>(kOutputSz, q, sycl::property_list{buffer_location(nnet::kOutputBufferLocation)});
#endif

        if (file_valid) {
            // Start always-run streaming kernel here, instead of inside a loop.
            q.single_task(MyProject{});

            // hls-fpga-machine-learning insert data

            // hls-fpga-machine-learning convert output

            // Print output from kernel and from prediction file.
            for (int i = 0; i < num_iterations; i++) {
                for (int j = 0; j < kOutLayerSize; j++) {
                    fout << outputs[i * kOutLayerSize + j] << " ";
                }
                fout << std::endl;
                if (i % CHECKPOINT == 0) {
                    std::cout << "Predictions" << std::endl;
                    // hls-fpga-machine-learning insert predictions
                    for (auto predval : predictions[i]) {
                        std::cout << predval << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "Quantized predictions" << std::endl;
                    // hls-fpga-machine-learning insert quantized
                    for (int j = 0; j < kOutLayerSize; j++) {
                        std::cout << outputs[i * kOutLayerSize + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        } else {
            std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                      << " invocations." << std::endl;
            q.single_task(MyProject{});
            // hls-fpga-machine-learning insert top-level-function
            // hls-fpga-machine-learning insert zero
            // hls-fpga-machine-learning convert output
            for (int i = 0; i < num_iterations; i++) {
                for (int j = 0; j < kOutLayerSize; j++) {
                    std::cout << outputs[i * kOutLayerSize + j] << " ";
                    fout << outputs[i * kOutLayerSize + j] << " ";
                }
                std::cout << std::endl;
                fout << std::endl;
            }
        }
        sycl::free(vals, q);
        sycl::free(outputs, q);
        fout.close();
        std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code.
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

        // Most likely the runtime couldn't find FPGA hardware!
        if (e.code().value() == CL_DEVICE_NOT_FOUND) {
            std::cerr << "If you are targeting an FPGA, please ensure that your "
                         "system has a correctly configured FPGA board.\n";
            std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
            std::cerr << "If you are targeting the FPGA emulator, compile with "
                         "-DFPGA_EMULATOR.\n";
        }
        std::terminate();
    }
    return 0;
}
