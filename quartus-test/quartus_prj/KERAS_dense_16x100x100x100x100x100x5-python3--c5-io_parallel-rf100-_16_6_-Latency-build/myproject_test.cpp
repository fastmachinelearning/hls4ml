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
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"

#define CHECKPOINT 5000

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
  int e = 0;

  int num_iterations = std::count(std::istreambuf_iterator<char>(fin),
                   std::istreambuf_iterator<char>(), '\n');

  if (fin.is_open() && fpr.is_open()) {
    //hls-fpga-machine-learning insert component-io
    inputdat input_1[num_iterations];
    outputdat layer13_out[num_iterations];
    std::vector<float> pr[num_iterations];
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      e++;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());

      current=strtok(cstr," ");
      while(current!=NULL) {
        pr[e].push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      std::vector<float>::const_iterator in_begin = in.cbegin();
      std::vector<float>::const_iterator in_end;
      in_end = in_begin + (N_INPUT_1_1);
      std::copy(in_begin, in_end, input_1[e].data);
      in_begin = in_end;

      //hls-fpga-machine-learning insert top-level-function
      ihc_hls_enqueue(&layer13_out[e], myproject, input_1[e]);
    }
    ihc_hls_component_run_all(myproject);
    for(int j = 0; j < e; j++) {
      //hls-fpga-machine-learning insert tb-output
      for(int i = 0; i < N_LAYER_12; i++) {
        fout << layer13_out[j].data[i] << " ";
      }
      fout << std::endl;
      if (j % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_12; i++) {
          std::cout << pr[j][i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        for(int i = 0; i < N_LAYER_12; i++) {
          std::cout << layer13_out[j].data[i] << " ";
        }
        std::cout << std::endl;
      }
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input with 4 invocations." << std::endl;
    num_iterations = 4;
    //hls-fpga-machine-learning insert zero
    inputdat input_1[num_iterations];
    outputdat layer13_out[num_iterations];

    for (int i = 0; i < num_iterations; i++) {
      //hls-fpga-machine-learning insert second-top-level-function
      std::fill_n(input_1[i].data, N_INPUT_1_1, 0.);
      ihc_hls_enqueue(&layer13_out[i], myproject, input_1[i]);
    }
    ihc_hls_component_run_all(myproject);

    for (int j = 0; j < num_iterations; j++) {
      //hls-fpga-machine-learning insert output
      for(int i = 0; i < N_LAYER_12; i++) {
        std::cout << layer13_out[j].data[i] << " ";
      }
      std::cout << std::endl;

      //hls-fpga-machine-learning insert tb-output
      for(int i = 0; i < N_LAYER_12; i++) {
        fout << layer13_out[j].data[i] << " ";
      }
      fout << std::endl;
    }
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
