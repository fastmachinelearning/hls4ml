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
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/prelu.h"
#include "firmware/nnet_utils/nnet_helpers.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

//hls-fpga-machine-learning insert bram
#include "firmware/weights/a2.h"

//hls-fpga-machine-learning insert declare weights
model_default_t a2[25];

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

CCS_MAIN(int argc, char *argv[])
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
            nnet::load_weights_from_txt<model_default_t, 25>(a2, "a2.txt");
        loaded_weights = true;
    }
#endif
  std::string iline;
  std::string pline;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }
//    std::cout << "    Input feature map size = " << in.size() << " Output predictions size = " << pr.size() << std::endl;

      //hls-fpga-machine-learning insert data
      input_t input_1[N_INPUT_1_1];
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1>(in, input_1);
      result_t layer2_out[N_INPUT_1_1];

      //hls-fpga-machine-learning insert top-level-function
      prelu(input_1,layer2_out,a2);

      for(int i = 0; i < N_INPUT_1_1; i++)
      {
	if(pr[i] != layer2_out[i])
	{
	 std::cout << "FAILURE" << std::endl;
	 std::cout << "Expected: " << pr[i] << " Actual: " << layer2_out[i].to_double() << std::endl;
	}
      }

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<result_t, N_INPUT_1_1>(layer2_out, fout);
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    input_t input_1[N_INPUT_1_1];
    nnet::fill_zero<input_t, N_INPUT_1_1>(input_1);
    result_t layer2_out[N_INPUT_1_1];

    //hls-fpga-machine-learning insert top-level-function
    prelu(input_1,layer2_out,a2);

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<result_t, N_INPUT_1_1>(layer2_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
