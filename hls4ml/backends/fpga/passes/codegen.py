import math
import numpy as np
from hls4ml.model.types import Source
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Dense, Conv1D, Conv2D

class GenerateConvIm2col(OptimizerPass):
    '''Generates tcode for im2col step of 1D/2d convolution'''

    def match(self, node):
        return isinstance(node, (Conv1D, Conv2D)) and node.model.config.get_config_value('IOType') == 'io_parallel'

    def transform(self, model, node):
        node_class = node.__class__.__name__
        if '1D' in node_class:
            self._generate_im2col_1d(node)
        elif '2D' in node_class:
            self._generate_im2col_2d(node)
        else:
            raise Exception(f'Cannot generate instructions for node {node.name} ({node_class})')

    def _generate_im2col_1d(self, node):
        code_str = node.model.config.backend.generate_conv1d_line_buffer_fn(
            node.get_attr('index'),
            node.get_attr('n_partitions'),
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            kernel=node.get_attr('filt_width'),
            stride=node.get_attr('stride_width'),
            pad=(node.get_attr('pad_left'), node.get_attr('pad_right')),
        )

        node.set_attr('line_buffer_codegen', Source(code_str))

    def _generate_im2col_2d(self, node):
        code_str = node.model.config.backend.generate_conv2d_line_buffer_fn(
            node.get_attr('index'),
            node.get_attr('n_partitions'),
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            node.get_input_variable().shape[2],
            kernel=(node.get_attr('filt_height'), node.get_attr('filt_width')),
            stride=(node.get_attr('stride_height'), node.get_attr('stride_width')),
            pad=(
                node.get_attr('pad_top'),
                node.get_attr('pad_bottom'),
                node.get_attr('pad_left'),
                node.get_attr('pad_right'),
            ),
        )

        node.set_attr('line_buffer_codegen', Source(code_str))

class GenerateUnrolledDenseResource(OptimizerPass):
    '''Generates C++ code for unrolled Dense resource'''

    def match(self, node):
        # Only apply to layers use that use Dense Matrix Multiplication
        # TODO - Extend (& test) for Conv1D / Separable Conv / Depthwise Conv / Recurrent layers
        layers_with_dense = (Dense, Conv2D)

        # Unrolled Dense mimicks the hardware implementation of Resource strategy -> apply after Resource optimizer
        weights_transposed = node.get_attr('_weights_transposed', False)

        # RF = 1 will optimize DSPs anyway, so no need to unroll code
        rf_gt_one = node.get_attr('reuse_factor') > 1

        # User requested unrolled implementation of Dense
        is_unrolled = node.get_attr('dense_resource_implementation', 'standard') == 'unrolled'

        return isinstance(node, layers_with_dense) and weights_transposed and rf_gt_one and is_unrolled

    def transform(self, model, node):
        code_str = self.__generate_unrolled_dense_resource(model, node)
        node.set_attr('unrolled_dense_resource_codegen', Source(code_str))
    
    def __generate_unrolled_dense_resource(self, model, node):
        """
        Generate a C++ function that mimics the Dense Resource implementation. Similar to Dense Resource, 3 cases are considered

        The HLS compiler produces suboptimal designs for Dense Resource when the weights processed by the same DSP are zero.
        Latency strategy can optimize zero mutiplications, Resource strategy, on the other hand, cannot.
        Furthermore, when all the weights in the same BRAM block are zero (e.g. due to model pruning), Vivado is unable to optimize it
        With this (and additional TCL scripts) zero BRAM are optimised

        Args:
            node: Layer to generate code for
        Returns:
            generated_code: Generated C++ function (string)
        """

        # Variable instantiation and function pragmas
        generated_code = (
            "template<class data_T, class res_T, typename CONFIG_T>\n"
            "class dense_unrolled_{index} : public DenseResourceUnrolled<data_T, res_T, CONFIG_T> {{\n"
            "    public:\n"
            "    static void dense_unrolled(\n"
            "    data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],\n"
            "    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],\n"
            "    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]\n"  
            "    ) {{\n"
            "        #pragma HLS pipeline II=CONFIG_T::reuse_factor\n"
            "\n"
            "        constexpr int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);\n"
            "        #pragma HLS function_instantiate variable=weights,biases\n"
            "        #pragma HLS ARRAY_RESHAPE variable=weights block factor=block_factor\n"    
            "        #pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM\n"
            "        #pragma HLS ARRAY_PARTITION variable=biases complete\n"
            "\n"        
            "        typename CONFIG_T::accum_t acc[CONFIG_T::n_out];\n"
            "        #pragma HLS ARRAY_PARTITION variable=acc complete\n"
            "\n"
            "        InitAccum:\n"
            "        for (int i = 0; i < CONFIG_T::n_out; i++) {{\n"
            "            #pragma HLS UNROLL\n"
            "            acc[i] = (typename CONFIG_T::accum_t) biases[i];\n"
            "        }}\n"
            "\n"
        ).format(index=node.index)
        
        # Unrolled multiplication, according to the three cases
        n_in, n_out = node.model.config.backend.get_layer_mult_size(node)
        reuse_factor = node.get_attr('reuse_factor')
        weights = node.weights['weight']
        if reuse_factor <= n_in:
            mult_code = self.__generate_unrolled_mult_code_rf_leq_nin(n_in, n_out, reuse_factor, weights)
        elif reuse_factor > n_in and reuse_factor % n_in == 0:
            mult_code = self.__generate_unrolled_mult_code_rf_gt_nin_rem0(n_in, n_out, reuse_factor, weights)
        else:
            # This case shouldn't happen if my understanding of RF is correct
            # The function fpga_backend._validate_reuse_factor() has assertion rf % n_in == 0 or rf < n_in
            raise Exception('Not implemented...')

        # Write output
        generated_code += mult_code + "\n"
        generated_code += (
            "        Result:\n"
            "        for (int i = 0; i < CONFIG_T::n_out; i++) {\n"
            "            #pragma HLS UNROLL\n"
            "            res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);\n"
            "        }\n"
            "    }\n"
            "};\n"
        )

        return generated_code

    def __generate_unrolled_mult_code_rf_leq_nin(self, n_in, n_out, reuse_factor, weights):
        # Function constants
        mult_factor = min(n_in, reuse_factor)
        block_factor = int(math.ceil(n_in * n_out / reuse_factor))
        mult_limit = int(math.ceil(n_in * n_out / mult_factor))
        mult_scale = mult_limit // n_out
        
        # Zero DSPs are the DSP blocks that always have zero input
        # In this case, it is the number of rows in the transposed and reshaped weight matrix
        # The new shape is (parallel_mult, reuse_factor)
        zeros = np.sum(~weights.data.reshape(block_factor, reuse_factor).any(1))

        # Generate unrolled multiplications
        mult_code = f"\t\t#pragma HLS ALLOCATION operation instances=mul limit={mult_limit - zeros}\n"
        mult_code += "\t\tMULT: {\n"
        mult_code += "\t\t\t#pragma HLS protocol\n"
        
        for ir in range(reuse_factor):
            acc_step = 0
            out_index = 0
            w_index = ir
            in_index = ir

            mult_code += f"\t\t\tM{ir}: {{\n"
            for _ in range(block_factor):
                if weights.data.flatten()[w_index] != 0:
                    mult_code += f"\t\t\t\tacc[{out_index}] += static_cast<typename CONFIG_T::accum_t>(CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[{in_index}], weights[{w_index}]));\n"
        
                w_index += reuse_factor
                in_index += reuse_factor
                if in_index >= n_in:
                    in_index = ir
                if acc_step + 1 >= mult_scale:
                    acc_step = 0
                    out_index += 1
                else:
                    acc_step += 1 
    
            mult_code += "\t\t\t}\n"
        
        mult_code += "\t\t}\n"

        return mult_code

    def __generate_unrolled_mult_code_rf_gt_nin_rem0(self, n_in, n_out, reuse_factor, weights):
        # Function constants
        mult_factor = min(n_in, reuse_factor)
        block_factor = int(math.ceil(n_in * n_out / reuse_factor))
        mult_limit = int(math.ceil(n_in * n_out / mult_factor))
        
        # Zero DSPs are the DSP blocks that always have zero input
        # In this case, it is the number of rows in the transposed and reshaped weight matrix
        # The new shape is (parallel_mult, reuse_factor)
        zeros = np.sum(~weights.data.reshape(block_factor, reuse_factor).any(1))
        
        # Generate out indices
        outidx = [0] * reuse_factor
        outstep = 0
        outscale = reuse_factor // n_in
        for ir in range(reuse_factor):
            outidx[ir] = outstep
            if (ir + 1) % n_in == 0:
                outstep += 1

        # Define variables
        in_index = 0

        # Generate unrolled multiplications
        mult_code = f"\t\t#pragma HLS ALLOCATION operation instances=mul limit={mult_limit - zeros}\n"
        mult_code += "\t\tMULT: {\n"
        mult_code += "\t\t\t#pragma HLS protocol\n"
        
        for ir in range(reuse_factor):
            w_index = ir
            out_index = outidx[ir]

            mult_code += f"\t\t\tM{ir}: {{\n"
            for _ in range(block_factor):
                if weights.data.flatten()[w_index] != 0:
                    mult_code += f"\t\t\t\tacc[{int(out_index)}] += static_cast<typename CONFIG_T::accum_t>(CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[{in_index}], weights[{w_index}]));\n"
        
                w_index += reuse_factor
                if w_index > n_in * n_out:
                    break
                out_index += outscale
            mult_code += "\t\t\t}\n"
        
            in_index += 1
            if in_index >= n_in:
                in_index = 0

        mult_code += "\t\t}\n"

        return mult_code

