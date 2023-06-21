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

#include "firmware/softmax.h"
#include "firmware/nnet_utils/nnet_helpers.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

//hls-fpga-machine-learning insert bram

//hls-fpga-machine-learning insert declare weights

#define CHECKPOINT 5
#define E_LIMIT 20

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
        loaded_weights = true;
    }
#endif

  std::cout << "Processing CHECKPOINT set to " << CHECKPOINT << std::endl;
  std::cout << "Processing E_LIMIT set to " << E_LIMIT << std::endl;
  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
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
      // vector of ac_fixed<16,6,true> type so that the output is being compared to the same type -- no rounding errors.
      std::vector<ac_fixed<16,6,true>> ac_pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
	ac_pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      ac_channel<input_t> input_1/*("input_1")*/;
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1>(in, input_1);
      ac_channel<result_t> layer2_out/*("layer2_out")*/;

      //hls-fpga-machine-learning insert top-level-function
      softmax(input_1,layer2_out);
//#if 0
      //if (e % CHECKPOINT == 0) {
        //std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
	result_t tmp = layer2_out[0];
        for(int i = 0; i < N_INPUT_1_1; i++) {
          if(tmp[i] != ac_pr[i])
          {
           std::cout << "Expected: " << ac_pr[i].to_double() << " Actual: " << tmp[i].to_double() << std::endl;
           return 1;
          }

        }
        //std::cout << std::endl;
        //std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
      //}
//#endif
      e++;

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<result_t, N_INPUT_1_1>(layer2_out, fout);

      if ((E_LIMIT > 0) && (e > E_LIMIT)) {
        std::cout << "Simulation stopping after " << E_LIMIT << " iterations" << std::endl;
        break;
      }
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    ac_channel<input_t> input_1/*("input_1")*/;
    nnet::fill_zero<input_t, N_INPUT_1_1>(input_1);
    ac_channel<result_t> layer2_out/*("layer2_out")*/;

    //hls-fpga-machine-learning insert top-level-function
    softmax(input_1,layer2_out);

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<result_t, N_INPUT_1_1>(layer2_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
