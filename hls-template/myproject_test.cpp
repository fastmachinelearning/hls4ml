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
#include <limits.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"

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

int main(int argc, char **argv)
{

  //hls-fpga-machine-learning insert data

  int retval = 0;

  result_t res_str[N_OUTPUTS] = {0};
  unsigned short size_in, size_out;
  myproject(data_str, res_str, size_in, size_out);

  for(int i=0; i<N_OUTPUTS; i++){
    std::cout << res_str[i] << " ";
  }
  std::cout << std::endl;


  // Validation
  std::ofstream results;
  results.open(LOG_FILE);
  for(int i=0; i<N_OUTPUTS; i++){
    results << res_str[i] << " ";
  }
  results << std::endl;
  results.close();

  std::string current_sim_log = get_exe_path() + LOG_FILE;
  std::string c_sim_log = get_exe_path() + "../../csim/build/" + LOG_FILE;

  std::string diff_cmd = "diff --brief -w ";
  diff_cmd += current_sim_log;
  diff_cmd += " ";
  diff_cmd += c_sim_log;

  retval = system(diff_cmd.c_str());
  if (retval != 0) {
      std::cout << "ERROR: test FAILED" << std::endl;
      std::cout << "ERROR:   current sim: " << current_sim_log << std::endl;
      std::cout << "ERROR:   C/C++   sim: " << c_sim_log << std::endl;
      retval = 1;
  } else {
      printf("INFO: test PASSED\n");
  }

  return retval;
}
