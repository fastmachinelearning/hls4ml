#ifndef BDT_H__
#define BDT_H__

#include "ap_fixed.h"

namespace BDT{

template<int n_nodes, int n_leaves, class input_t, class score_t, class threshold_t>
struct Tree {
	int feature[n_nodes];
	threshold_t threshold[n_nodes];
	score_t value[n_nodes];
	int children_left[n_nodes];
	int children_right[n_nodes];
	int parent[n_nodes];

	score_t decision_function(input_t x) const{
		#pragma HLS pipeline II = 1
#pragma HLS RESOURCE variable=feature core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=threshold core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=value core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=children_left core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=children_right core=ROM_nP_LUTRAM


		bool comparison[n_nodes];
		bool activation[n_nodes];
		bool activation_leaf[n_leaves];
		score_t value_leaf[n_leaves];

		#pragma HLS ARRAY_PARTITION variable=comparison
		#pragma HLS ARRAY_PARTITION variable=activation
		#pragma HLS ARRAY_PARTITION variable=activation_leaf
		#pragma HLS ARRAY_PARTITION variable=value_leaf

		// Execute all comparisons
		Compare: for(int i = 0; i < n_nodes; i++){
			#pragma HLS unroll
			// Only non-leaf nodes do comparisons
			if(x[feature[i]] != -2){ // -2 means is a leaf (at least for sklearn)
				comparison[i] = x[feature[i]] <= threshold[i];
			}else{
				comparison[i] = true;
			}
		}

		// Determine node activity for all nodes
		int iLeaf = 0;
		Activate: for(int i = 0; i < n_nodes; i++){
			#pragma HLS unroll
			// Root node is always active
			if(i == 0){
				activation[i] = true;
			}else{
				// If this node is the left child of its parent
				if(i == children_left[parent[i]]){
					activation[i] = comparison[parent[i]] && activation[parent[i]];
				}else{ // Else it is the right child
					activation[i] = !comparison[parent[i]] && activation[parent[i]];
				}
			}
			// Skim off the leaves
			if(children_left[i] == -1){ // is a leaf
				activation_leaf[iLeaf] = activation[i];
				value_leaf[iLeaf] = value[i];
				iLeaf++;
			}
		}

		// Set each bit of addr to the corresponding leaf activation
		ap_uint<n_leaves> active_leaf_addr;
		Select: for(int i = 0; i < n_leaves; i++){
			#pragma HLS unroll
			active_leaf_addr[i] = activation_leaf[i];
		}
		score_t y;
		switch (active_leaf_addr){
		case 1 : y = value_leaf[0]; break;
		case 2 : y = value_leaf[1]; break;
		case 4 : y = value_leaf[2]; break;
		case 8 : y = value_leaf[3]; break;
		case 16 : y = value_leaf[4]; break;
		case 32 : y = value_leaf[5]; break;
		case 64 : y = value_leaf[6]; break;
		case 128 : y = value_leaf[7]; break;
		default : y = 0;
		}
		return y;
	}
};

template<int n_trees, int n_nodes, int n_leaves, class input_t, class score_t, class threshold_t>
struct BDT{
	Tree<n_nodes, n_leaves, input_t, score_t, threshold_t> trees[n_trees];

	score_t decision_function(input_t x) const{
		score_t score = 0;
		Trees: for(int i = 0; i < n_trees; i++){
			#pragma HLS UNROLL
			score += trees[i].decision_function(x);
		}
		return score;
	}
};

}
#endif
