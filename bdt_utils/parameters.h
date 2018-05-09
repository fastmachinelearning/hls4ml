#ifndef BDT_PARAMS_H__
#define BDT_PARAMS_H__

#include "ap_fixed.h"

static const int n_trees = 100;
static const int n_nodes = 15;
static const int n_leaves = 8;
static const int n_features = 4; // Number of features

typedef ap_fixed<18,9> input_t; // Input features type
typedef input_t input_arr_t[n_features];
typedef ap_fixed<18, 9> score_t; // Type for tree score
typedef input_t threshold_t;

// TODO make this work
// Struct for input data, allowing different types
typedef struct input_struct_t{
	ap_fixed<18, 9> x0;
	ap_fixed<18, 9> x1;
	ap_fixed<17, 8> x2;
	ap_uint<2> x3;

	template<unsigned N, unsigned M> ap_fixed<N,M> &operator[](int i){
		switch(i){
		case 0 : return x0;
		case 1 : return x1;
		case 2 : return x2;
		case 3 : return x3;
		default : std::cout << "No index " << i << std::endl; return x0;
		}
	}

} input_struct_t;

#endif
