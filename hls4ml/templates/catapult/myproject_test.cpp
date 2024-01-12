#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

static std::string s_weights_dir;

const char *get_weights_dir() {
  return s_weights_dir.c_str();  
}

#include "firmware/myproject.h"
#include "nnet_utils/nnet_helpers.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

//hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

#ifndef RANDOM_FRAMES
#define RANDOM_FRAMES 1
#endif

//hls-fpga-machine-learning insert declare weights

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

CCS_MAIN(int argc, char *argv[])
{
  if (argc < 4) {
    std::cerr << "Error - too few arguments" << std::endl;
    std::cerr << "Usage: " << argv[0] << " <weights_dir> <tb_input_features> <tb_output_predictions>" << std::endl;
    std::cerr << "Where:    <weights_dir>           - string pathname to directory containing wN.txt and bN.txt files" << std::endl;
    std::cerr << "          <tb_input_features>     - string pathname to tb_input_features.dat" << std::endl;
    std::cerr << "          <tb_output_predictions> - string pathname to tb_output_predictions.dat" << std::endl;
    CCS_RETURN(-1);
  }
  s_weights_dir = argv[1];
  std::string tb_in(argv[2]);
  std::string tb_out(argv[3]);
  std::cout << "  Weights directory: " << s_weights_dir << std::endl;
  std::cout << "  Test Feature Data: " << tb_in << std::endl;
  std::cout << "  Test Predictions : " << tb_out << std::endl;

  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");
  std::cout<<"Number of Frames Passed from the tcl= " << RANDOM_FRAMES << std::endl;

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
  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0)
        std::cout << "Processing input " << e << std::endl;
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

      //hls-fpga-machine-learning insert top-level-function

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        // hls-fpga-machine-learning insert predictions
        std::cout << "Quantized predictions" << std::endl;
        // hls-fpga-machine-learning insert quantized
      }
      e++;

      //hls-fpga-machine-learning insert tb-output
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using so feeding random values" << std::endl;

	if(RANDOM_FRAMES>0)
	{
	for (unsigned int k=0; k<RANDOM_FRAMES; k++){
    //hls-fpga-machine-learning insert random

    //hls-fpga-machine-learning insert top-level-function

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
	}
	}
	else{
	//hls-fpga-machine-learning insert zero

    //hls-fpga-machine-learning insert top-level-function

    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
	}
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
