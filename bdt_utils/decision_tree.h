#ifndef MYPROJECT_H__
#define MYPROJECT_H__

#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_cint.h"

typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> output_t;

output_t tree(input_t x[4]);//, output_t y[1]); //bool activations[15]);

#endif
