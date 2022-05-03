//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cctype>

#include "firmware/parameters.h"
#include "firmware/myproject.h"

#define CHECKPOINT 5000

// This function is written to avoid stringstream, which is
// not supported in cosim 20.1, and because strtok
// requires a const_cast or allocation to use with std::strings.
// This function returns the next float (by argument) at position pos,
// updating pos. True is returned if conversion done, false if the string
// has ended, and std::invalid_argument exception if the sting was bad.
bool nextToken(const std::string& str, std::size_t& pos, float& val)
{
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

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");


  std::string RESULTS_LOG = "tb_data/results.log";
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;

  std::vector<input_data> inputs;
  std::vector<output_data> outputs;

  if (fin.is_open() && fpr.is_open()) {
    std::vector<std::vector<float> > predictions;
    unsigned int num_iterations = 0;
    for (; std::getline(fin,iline) && std::getline (fpr,pline); num_iterations++) {
      if (num_iterations % CHECKPOINT == 0) {
        std::cout << "Processing input "  << num_iterations << std::endl;
      }

      std::vector<float> in;
      std::vector<float> pr;
      float current;

      std::size_t pos = 0;
      while(nextToken(iline, pos, current)) {
        in.push_back(current);
      }

      pos = 0;
      while(nextToken(pline, pos, current)) {
        pr.push_back(current);
      }

      //hls-fpga-machine-learning insert data
      predictions.push_back(std::move(pr));
    }

    // Do this separately to avoid vector reallocation
    //hls-fpga-machine-learning insert top-level-function

    //hls-fpga-machine-learning insert run


    for(int j = 0; j < num_iterations; j++) {
      //hls-fpga-machine-learning insert tb-output
      if (j % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
      }
    }
    fin.close();
    fpr.close();
  } else {
    const unsigned int num_iterations = 10;
    std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations << " invocations." << std::endl;
    //hls-fpga-machine-learning insert zero

    //hls-fpga-machine-learning insert top-level-function

    //hls-fpga-machine-learning insert run

    for (int j = 0; j < num_iterations; j++) {
      //hls-fpga-machine-learning insert output

      //hls-fpga-machine-learning insert tb-output
    }
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
