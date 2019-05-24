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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"

// In Vivado HLS, C and RTL simulations have different working directories,
// let's call them generically WORK_DIR.
//
// - For C simulations,   WORK_DIR is "myproject_prj/solution1/csim/build"
// - For RTL simulations, WORK_DIR is "myproject_prj/solution1/sim/wrapc"
//
// In both the cases, during the simulation, this testbench saves the log file
// in the current (either C or RTL) WORK_DIR.
//
// For validation purposes, we consider the C-simulation results as the golden
// reference. The validation compares (diff) the current log file (either from
// C or RTL simulation) with the golden reference which always sits in
// "WORK_DIR/../csim/build".
//
// NOTES:
// - The C simulations always pass the validation, because the log file is
//   compared with itself (they both sits in "WORK_DIR/../csim/build").
// - The C simulation has to be run before the RTL simulation.
//

#ifdef VALIDATION
#define LOG_FILE "results.log"

std::string get_exe_path()
{
  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) == NULL) {
    return "./";
  } else {
    return (std::string(cwd) + "/");
  }
}
#endif

int main(int argc, char **argv)
{
  int retval = 0;

  //hls-fpga-machine-learning insert data

  //hls-fpga-machine-learning insert top-level-function

  //hls-fpga-machine-learning insert output

#ifdef VALIDATION
  // The results are saved in the current working directory.
  std::ofstream results;
  results.open(LOG_FILE);
  for(int i=0; i<N_OUTPUTS; i++){
    results << layer_out[i] << " ";
  }
  results << std::endl;
  results.close();

  const std::string WORK_DIR = get_exe_path();
  const std::string GOLD_DIR = get_exe_path() + "../../csim/build/";

  std::string diff_cmd = "diff --brief -w ";
  diff_cmd += WORK_DIR + LOG_FILE;
  diff_cmd += " ";
  diff_cmd += GOLD_DIR + LOG_FILE;

  retval = system(diff_cmd.c_str());
  if (retval != 0) {
      std::cout << "ERROR: test FAILED" << std::endl;
      std::cout << "ERROR:   current sim: " << (WORK_DIR + LOG_FILE) << std::endl;
      std::cout << "ERROR:   golden  sim: " << (GOLD_DIR + LOG_FILE) << std::endl;
      retval = 1;
  } else {
      printf("INFO: test PASSED\n");
  }
#endif

  return retval;
}
