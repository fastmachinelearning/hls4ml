import math

import numpy as np

from hls4ml.model.layers import GRU, LSTM, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import Source


class GenerateUnrolledDenseResource(OptimizerPass):
    '''Generates C++ code for unrolled Dense resource'''

    def match(self, node):
        # Only apply to layers use that use Dense Matrix Multiplication
        # TODO - Extend (& test) for Separable Conv / Depthwise Conv / Recurrent layers
        layers_with_dense = (Dense, Conv1D, Conv2D, LSTM, GRU)

        # Unrolled Dense mimics the hardware implementation of Resource strategy -> apply after Resource optimizer
        weights_transposed = node.get_attr('_weights_transposed', False)

        # RF = 1 will optimize DSPs anyway, so no need to unroll code
        rf_gt_one = node.get_attr('reuse_factor', 1) > 1

        # User requested unrolled implementation of Dense
        is_unrolled = node.get_attr('strategy', 'latency') == 'resource_unrolled'

        return isinstance(node, layers_with_dense) and weights_transposed and rf_gt_one and is_unrolled

    def transform(self, model, node):
        if isinstance(node, (LSTM, GRU)):
            n_in, n_out, n_in_recr, n_out_recr = node.model.config.backend.get_layer_mult_size(node)

            reuse_factor = node.get_attr('reuse_factor')
            weights = node.weights['weight']
            code_str = self._generate_unrolled_function(n_in, n_out, reuse_factor, weights, str(node.index) + '_1')
            code_str = self._add_backend_specific_pragmas_to_generated_code(code_str, model.config.backend)
            node.set_attr('resource_unrolled_dense_codegen_1', Source(code_str))

            recr_reuse_factor = node.get_attr('recurrent_reuse_factor')
            recr_weights = node.weights['recurrent_weight']
            code_str = self._generate_unrolled_function(
                n_in_recr, n_out_recr, recr_reuse_factor, recr_weights, str(node.index) + '_2'
            )
            code_str = self._add_backend_specific_pragmas_to_generated_code(code_str, model.config.backend)
            node.set_attr('resource_unrolled_dense_codegen_2', Source(code_str))

        else:
            n_in, n_out = node.model.config.backend.get_layer_mult_size(node)
            reuse_factor = node.get_attr('reuse_factor')
            weights = node.weights['weight']

            code_str = self._generate_unrolled_function(n_in, n_out, reuse_factor, weights, node.index)
            code_str = self._add_backend_specific_pragmas_to_generated_code(code_str, model.config.backend)
            node.set_attr('resource_unrolled_dense_codegen', Source(code_str))

    def _generate_unrolled_function(self, n_in, n_out, reuse_factor, weights, function_suffix):
        """
        Generate a C++ function that mimics the Dense Resource implementation.

        The HLS compiler produces suboptimal designs for Dense Resource when the weights processed by the same DSP are zero.
        Latency strategy can optimize zero multiplications
        Resource strategy, on the other hand, cannot.
        When all the weights in the same BRAM block are zero, Vivado is unable to optimize it
        With this (and additional TCL scripts) zero BRAM are optimized

        Args:
            node: Layer to generate code for
        Returns:
            generated_code: Generated C++ function (string)
        """

        # Variable instantiation and function pragmas
        generated_code = (
            'template<class data_T, class res_T, typename CONFIG_T>\n'
            'class dense_resource_unrolled_{suffix} : public DenseKernel<data_T, res_T, CONFIG_T> {{{{\n'
            '    public:\n'
            '    static void dense(\n'
            '    data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],\n'
            '    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],\n'
            '    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]\n'
            '    ) {{{{\n'
            '        #pragma HLS pipeline II=CONFIG_T::reuse_factor\n'
            '\n'
            '        constexpr int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);\n'
            '        #pragma HLS ARRAY_RESHAPE variable=weights block factor=block_factor\n'
            '        {{weights_resource_pragma}}\n'
            '        #pragma HLS ARRAY_PARTITION variable=biases complete\n'
            '\n'
            '        typename CONFIG_T::accum_t acc[CONFIG_T::n_out];\n'
            '        #pragma HLS ARRAY_PARTITION variable=acc complete\n'
            '\n'
            '        InitAccum:\n'
            '        for (int i = 0; i < CONFIG_T::n_out; i++) {{{{\n'
            '            #pragma HLS UNROLL\n'
            '            acc[i] = (typename CONFIG_T::accum_t) biases[i];\n'
            '        }}}}\n'
            '\n'
        ).format(suffix=function_suffix)

        # Unrolled multiplication, according to the three cases
        if reuse_factor <= n_in:
            mult_code = self._generate_unrolled_mult_code_rf_leq_nin(n_in, n_out, reuse_factor, weights)
        elif reuse_factor > n_in and reuse_factor % n_in == 0:
            mult_code = self._generate_unrolled_mult_code_rf_gt_nin_rem0(n_in, n_out, reuse_factor, weights)
        else:
            # This case shouldn't happen if my understanding of RF is correct
            # The function fpga_backend._validate_reuse_factor() has assertion rf % n_in == 0 or rf < n_in
            raise Exception('Not implemented...')

        # Write output
        generated_code += mult_code + '\n'
        generated_code += (
            '        Result:\n'
            '        for (int i = 0; i < CONFIG_T::n_out; i++) {{\n'
            '            #pragma HLS UNROLL\n'
            '            res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);\n'
            '        }}\n'
            '    }}\n'
            '}};\n'
        )

        return generated_code

    def _generate_unrolled_mult_code_rf_leq_nin(self, n_in, n_out, reuse_factor, weights):
        # Function constants
        mult_factor = min(n_in, reuse_factor)
        block_factor = int(math.ceil(n_in * n_out / reuse_factor))
        mult_limit = int(math.ceil(n_in * n_out / mult_factor))
        mult_scale = mult_limit // n_out

        # Zero DSPs are the DSP blocks that always have zero input
        # In this case, it is the number of rows in the transposed and reshaped weight matrix
        # The new shape is (parallel_mult, reuse_factor)
        zeros = np.sum(~weights.data.reshape(block_factor, reuse_factor).any(1))

        # Used to pad the code to make it human-readable
        indent = '    '

        # Generate unrolled multiplications
        mult_code = f'{indent*2}#pragma HLS ALLOCATION operation instances=mul limit={mult_limit - zeros}\n'
        mult_code += f'{indent*2}MULT: {{{{\n'

        for ir in range(reuse_factor):
            acc_step = 0
            out_index = 0
            w_index = ir
            in_index = ir

            mult_code += f'{indent*3}M{ir}: {{{{\n'
            for _ in range(block_factor):
                if weights.data.flatten()[w_index] != 0:
                    mult_code += (
                        f'{indent*4}acc[{out_index}] += '
                        'static_cast<typename CONFIG_T::accum_t>'
                        '(CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::'
                        f'product(data[{in_index}], weights[{w_index}]));\n'
                    )

                w_index += reuse_factor
                in_index += reuse_factor
                if in_index >= n_in:
                    in_index = ir
                if acc_step + 1 >= mult_scale:
                    acc_step = 0
                    out_index += 1
                else:
                    acc_step += 1

            mult_code += f'{indent*3}}}}}\n'

        mult_code += f'{indent*2}}}}}\n'

        return mult_code

    def _generate_unrolled_mult_code_rf_gt_nin_rem0(self, n_in, n_out, reuse_factor, weights):
        # Function constants
        mult_factor = min(n_in, reuse_factor)
        block_factor = int(math.ceil(n_in * n_out / reuse_factor))
        mult_limit = int(math.ceil(n_in * n_out / mult_factor))

        # Zero DSPs are the DSP blocks that always have zero input
        # In this case, it is the number of rows in the transposed and reshaped weight matrix
        # The new shape is (parallel_mult, reuse_factor)
        zeros = np.sum(~weights.data.reshape(block_factor, reuse_factor).any(1))

        # Used to pad the code to make it human-readable
        indent = '    '

        # Generate out indices
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
        mult_code = f'{indent*2}#pragma HLS ALLOCATION operation instances=mul limit={mult_limit - zeros}\n'
        mult_code += f'{indent*2}MULT: {{{{\n'

        for ir in range(reuse_factor):
            w_index = ir
            out_index = outidx[ir]

            mult_code += f'{indent*3}M{ir}: {{{{\n'
            for _ in range(block_factor):
                if weights.data.flatten()[w_index] != 0:
                    mult_code += (
                        f'{indent*4}acc[{int(out_index)}] += '
                        'static_cast<typename CONFIG_T::accum_t>'
                        '(CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::'
                        f'product(data[{in_index}], weights[{w_index}]));\n'
                    )

                w_index += reuse_factor
                if w_index > n_in * n_out:
                    break
                out_index += outscale
            mult_code += f'{indent*3}}}}}\n'

            in_index += 1
            if in_index >= n_in:
                in_index = 0

        mult_code += f'{indent*2}}}}}\n'

        return mult_code

    def _add_backend_specific_pragmas_to_generated_code(self, code, backend):
        if backend.name == 'Vivado':
            weights_resource_pragma = '#pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM'
        elif backend.name == 'Vitis':
            weights_resource_pragma = '#pragma HLS BIND_STORAGE variable=weights type=ROM_NP impl=BRAM'
        else:
            raise Exception(f'Unexpected backend {backend.name} in GenerateUnrolledDenseResource optimizer.')

        code = code.format(weights_resource_pragma=weights_resource_pragma)

        return code
