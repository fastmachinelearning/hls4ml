#ifndef MYPROJECT_H__
#define MYPROJECT_H__

#include "parameters.h"
#include "ap_int.h"
#include "ap_fixed.h"


template<int n_nodes, int n_leaves>
class Tree{
private:
	BDT::tree_params_t<n_nodes> tree;

public:

	Tree(struct BDT::tree_params_t<n_nodes> tree)
	: tree(tree){}

	BDT::score_t decision_function(BDT::input_arr_t x){
		#pragma HLS pipeline II = 1

		bool comparison[n_nodes];
		bool activation[n_nodes];
		bool activation_leaf[n_leaves];
		double value_leaf[n_leaves];

		// Execute all comparisons
		for(int i = 0; i < n_nodes; i++){
			#pragma HLS unroll
			// Only non-leaf nodes do comparisons
			if(x[tree.feature[i]] != -2){ // -2 means is a leaf (at least for sklearn)
				comparison[i] = x[tree.feature[i]] <= tree.threshold[i];
			}else{
				comparison[i] = true;
			}
		}

		// Determine node activity for all nodes
		int iLeaf = 0;
		for(int i = 0; i < n_nodes; i++){
			#pragma HLS unroll
			// Root node is always active
			if(i == 0){
				activation[i] = true;
			}else{
				// If this node is the left child of its parent
				if(i == tree.children_left[tree.parent[i]]){
					activation[i] = comparison[tree.parent[i]] && activation[tree.parent[i]];
				}else{ // Else it is the right child
					activation[i] = !comparison[tree.parent[i]] && activation[tree.parent[i]];
				}
			}
			// Skim off the leaves
			if(tree.children_left[i] == -1){ // is a leaf
				activation_leaf[iLeaf] = activation[i];
				value_leaf[iLeaf] = tree.value[i];
				iLeaf++;
			}
		}

		// Set each bit of addr to the corresponding leaf activation
		ap_uint<n_leaves> active_leaf_addr;
		for(int i = 0; i < n_leaves; i++){
			#pragma HLS unroll
			active_leaf_addr[i] = activation_leaf[i];
		}
		BDT::score_t y;
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

	// TODO as above but with the struct so that features can have differing types
	//BDT::score_t decision_function(BDT::input_struct_t x);

};

#endif
