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

#include "firmware/softsign.h"
#include "firmware/nnet_utils/nnet_helpers.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

//hls-fpga-machine-learning insert bram
#include "firmware/weights/w2.h"
#include "firmware/weights/b2.h"

//hls-fpga-machine-learning insert declare weights
model_default_t w2[9];
model_default_t b2[1];

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
            nnet::load_weights_from_txt<model_default_t, 9>(w2, "w2.txt");
            nnet::load_weights_from_txt<model_default_t, 1>(b2, "b2.txt");
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
      ac_channel<input_t> conv2d_input/*("conv2d_input")*/;
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(in, conv2d_input);
      ac_channel<result_t> layer3_out/*("layer3_out")*/;

      //hls-fpga-machine-learning insert top-level-function
      softsign(conv2d_input,layer3_out,w2,b2);

      for(int i = 0; i < OUT_HEIGHT_2*OUT_WIDTH_2; i++)
      {
	if(fabs(pr[i]-layer3_out[i][0].to_double()) > 0.003)
	{
	 std::cout << "FAILURE" << std::endl;
       	 std::cout << "Expected: " << pr[i] << " Actual: " << layer3_out[i][0].to_double() << std::endl;
	 return 1;
	}
      }

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<result_t, OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2>(layer3_out, fout);
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    ac_channel<input_t> conv2d_input/*("conv2d_input")*/;
    nnet::fill_zero<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(conv2d_input);
    ac_channel<result_t> layer3_out/*("layer3_out")*/;

    //hls-fpga-machine-learning insert top-level-function
    softsign(conv2d_input,layer3_out,w2,b2);

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<result_t, OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2>(layer3_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
