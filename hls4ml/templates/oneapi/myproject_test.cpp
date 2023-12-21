#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>

#include "exception_handler.hpp"
// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000


int main(int argc, char **argv) {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    // make sure the device supports USM host allocations
    if (!device.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "This design must either target a board that supports USM "
                   "Host/Shared allocations, or IP Component Authoring. "
                << std::endl;
      std::terminate();
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;


    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;

    std::vector<input_data_t> inputs;
    std::vector<output_data_t> outputs;

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;
        unsigned int num_iterations = 0;
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
            if (in.size() != N_INPUT_1_1) {
                throw std::runtime_error("The input size does not match");
            }
            if (pr.size() != N_LAYER_11) {
                throw std::runtime_error("The output size does not match");
            }

            // hls-fpga-machine-learning insert data
            inputs.emplace_back();
            std::copy(in.cbegin(), in.cend(), inputs.back().begin());
            outputs.emplace_back();
            predictions.push_back(std::move(pr));
        }
        // Do this separately to avoid vector reallocation
        // hls-fpga-machine-learning insert top-level-function
        for(int i = 0; i < num_iterations; i++) {
            InPipe::write(q, inputs[i]);
            q.single_task(MyProject{});  // once or once for each
        }
        q.wait();

        for (int j = 0; j < num_iterations; j++) {
            // hls-fpga-machine-learning insert tb-output
            outputs[j] = OutPipe::read(q);
            for(int i = 0; i < N_LAYER_11; i++) {
              fout << outputs[j][i] << " ";
            }
            fout << std::endl;
            if (j % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for(int i = 0; i < N_LAYER_11; i++) {
                  std::cout << predictions[j][i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                for(int i = 0; i < N_LAYER_11; i++) {
                  std::cout << outputs[j][i] << " ";
                }
                std::cout << std::endl;
            }
        }
        fin.close();
        fpr.close();
    } else {
        const unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;
        // hls-fpga-machine-learning insert zero
        for(int i = 0; i < num_iterations; i++) {
            inputs.emplace_back();
            outputs.emplace_back();
            outputs.back().fill(0.0);
        }

        // hls-fpga-machine-learning insert top-level-function
        for(int i = 0; i < num_iterations; i++) {
            InPipe::write(q, inputs[i]);
            q.single_task(MyProject{});
        }
        q.wait();

        for (int j = 0; j < num_iterations; j++) {
            // hls-fpga-machine-learning insert output
            outputs[j] = OutPipe::read(q);
            for(int i = 0; i < N_LAYER_11; i++) {
              std::cout << outputs[j][i] << " ";
            }
            std::cout << std::endl;

            // hls-fpga-machine-learning insert tb-output
            for(int i = 0; i < N_LAYER_11; i++) {
              fout << outputs[j][i] << " ";
            }
            fout << std::endl;
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
