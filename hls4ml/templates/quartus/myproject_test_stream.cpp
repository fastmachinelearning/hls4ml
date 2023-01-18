#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

// This function is written to avoid stringstream, which is
// not supported in cosim 20.1, and because strtok
// requires a const_cast or allocation to use with std::strings.
// This function returns the next float (by argument) at position pos,
// updating pos. True is returned if conversion done, false if the string
// has ended, and std::invalid_argument exception if the sting was bad.
bool nextToken(const std::string &str, std::size_t &pos, float &val) {
    while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos]))) {
        pos++;
    }
    if (pos >= str.size()) {
        return false;
    }
    std::size_t offset = 0;
    val = std::stof(str.substr(pos), &offset);
    pos += offset;
    return true;
}

int main(int argc, char **argv) {
    // Load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::string iline;

    // Load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    std::string pline;

    // Output log
    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;

        unsigned int iteration = 0;
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Processing input " << iteration << std::endl;
            }

            // hls-fpga-machine learning instantiate inputs and outputs

            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::size_t pos = 0;
            while (nextToken(iline, pos, current)) {
                in.push_back(current);
            }

            pos = 0;
            while (nextToken(pline, pos, current)) {
                pr.push_back(current);
            }

            // hls-fpga-machine-learning insert data

            predictions.push_back(std::move(pr));

            // hls-fpga-machine-learning insert top-level-function

            // hls-fpga-machine-learning insert run

            // hls-fpga-machine-learning convert output

            // hls-fpga-machine-learning insert tb-output

            if (iteration % CHECKPOINT == 0) {
                std::cout << "Python Predictions" << std::endl;
                // hls-fpga-machine-learning print predictions

                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
            }

            iteration++;
        }

        fin.close();
        fpr.close();

    } else {
        const unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        for (int iteration = 0; iteration < num_iterations; iteration++) {
            // hls-fpga-machine learning instantiate inputs and outputs

            // hls-fpga-machine-learning insert zero

            // hls-fpga-machine-learning insert top-level-function

            // hls-fpga-machine-learning insert run

            // hls-fpga-machine-learning convert output

            // hls-fpga-machine-learning insert tb-output

            if (iteration % CHECKPOINT == 0) {
                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
            }
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
