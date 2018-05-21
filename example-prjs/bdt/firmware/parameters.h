#ifndef BDT_PARAMS_H__
#define BDT_PARAMS_H__

#include "ap_fixed.h"

static const int n_trees = 100;
static const int max_depth = 3;
static const int n_features = 4;
typedef ap_fixed<18,8> input_t;
typedef input_t input_arr_t[n_features];
typedef ap_fixed<18,8> score_t;
typedef input_t threshold_t;

#endif