/**
 * @brief myproject_host.cpp
 *
 * This file is a stand-alone C++ program that can be used to run inference of an hls4ml
 * model with Coyote. The alternative way is to use the CoyoteOverlay from Python.
 * Both of these rely on the CoyoteInference class from the host_libs.hpp file.
 * The format of this script is largely similar to myproject_test.cpp (i.e. it reads the
 * inputs and outputs from some files and runs inference), but adapted to run on an FPGA.
 */

#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "defines.h"
#include "host_libs.hpp"

#include <boost/program_options.hpp>

std::string default_path("../../tb_data/");

int main(int argc, char **argv) {
    std::string data_path;
    unsigned int batch_size;

    boost::program_options::options_description runtime_options("Coyote hls4ml run-time options");
    runtime_options.add_options()
        ("batch_size,b", boost::program_options::value<unsigned int>(&batch_size)->default_value(1), "Inference batch size")
        ("data_path,p", boost::program_options::value<std::string>(&data_path)->default_value(default_path), "Path to tb_data folder with input/output features for validation");
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);

    // hls-fpga-machine-learning insert I/O size

    CoyoteInference model(batch_size, in_size, out_size);

    std::string iline;
    std::string pline;
    std::ifstream fin(data_path + "/tb_input_features.dat");
    std::ifstream fpr(data_path + "/csim_results.log");
    
    if (fin.is_open() && fpr.is_open()) {
        int cnt = 0;
        int total_batches = 0;
        double avg_latency = 0;
        double avg_throughput = 0;
        std::vector<std::vector<float>> labels;

        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            // Read inputs and outputs from tb_data folder
            char *current;
            std::vector<float> in, pr;

            char *cstr = const_cast<char *>(iline.c_str());
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            
            // Set model data for the i-th point in the batch
            model.set_data(&in[0], cnt);
            labels.push_back(pr);
            cnt++;

            // If batch is full, run inference, measuring time
            if (cnt == batch_size) {
                model.flush();

                auto begin_time = std::chrono::high_resolution_clock::now();
                model.predict();
                auto end_time = std::chrono::high_resolution_clock::now();
                double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();                
                avg_latency += (time / 1e3);
                avg_throughput += (batch_size / (time * 1e-9));
                
                // Functional correctness
                for (int i = 0; i < batch_size; i++) { 
                    float *pred = model.get_predictions(i);
                    for (int j = 0; j < out_size; j++) {
                        assert(int(10000.0 * labels[i][j]) == int(10000.0 * pred[j])); 
                    } 
                }
                
                // Reset for next batch
                total_batches++;
                labels.clear();
                cnt = 0;
            }

        }

        std::cout << "Batches processed: " << total_batches << std::endl;
        std::cout << "Average latency: " << avg_latency / (double) total_batches << " us" << std::endl;
        std::cout << "Average throughput: " << avg_throughput / (double) total_batches << " inferences/s" << std::endl;

        fin.close();
        fpr.close();
    } else {
        std::cout << "Couldn't open input/output file; make sure data_path is set correctly!" << std::endl;
    }
    
    return EXIT_SUCCESS;
}
