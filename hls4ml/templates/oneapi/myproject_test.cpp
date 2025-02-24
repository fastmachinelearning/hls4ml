#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

#if (__INTEL_CLANG_COMPILER < 20250000)
#include <sycl/ext/intel/prototype/interfaces.hpp>
#endif

#include "exception_handler.hpp"
// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

int main(int argc, char **argv) {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
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

    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> predictions;
        unsigned int num_iterations = 0;
        
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
                for (int j = 0; j < kOutLayerSize /* defined in convert output */; j++) {
                    std::cout << outputs[i * kOutLayerSize + j] << " ";
                }
                std::cout << std::endl;
            }
        }
        delete[] vals;
        delete[] outputs;
        fin.close();
        fpr.close();
    } else {
        constexpr unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        // hls-fpga-machine-learning insert top-level-function

        // hls-fpga-machine-learning insert zero
        q.single_task(MyProject{});
        // hls-fpga-machine-learning convert output
        for (int i = 0; i < num_iterations; i++) {
            for (int j = 0; j < kOutLayerSize /* defined in convert output */; j++) {
                std::cout << outputs[i * kOutLayerSize + j] << " ";
                fout << outputs[i * kOutLayerSize + j] << " ";
            }
            std::cout << std::endl;
            fout << std::endl;
        }
        delete[] outputs;
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
