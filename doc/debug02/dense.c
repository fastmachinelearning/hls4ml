#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N_IN 16
#define N_OUT 8

#define RF 32

#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

void dense(
    float data[N_IN],
    float res[N_OUT],
    float weights[N_IN*N_OUT],
    float biases[N_OUT])
{
    float cache;
    float mult[N_IN*N_OUT];
    float acc[N_OUT];

    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < N_IN; ii++) {
        cache = data[ii];
        Product2: for(int jj = 0; jj < N_OUT; jj++) {
            int index = ii*N_OUT+jj;
            mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < N_OUT; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < N_IN; ii++) {
        Accum2: for(int jj = 0; jj < N_OUT; jj++) {
            int index = ii*N_OUT+jj;
            acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < N_OUT; ires++){
        res[ires] = (float) (acc[ires]);
    }
}

void dense_v4(
    float data[N_IN],
    float res[N_OUT],
    float weights[N_IN*N_OUT],
    float biases[N_OUT])
{

    const int totals_multipliers = N_IN*N_OUT;
    const int multiplier_limit = DIV_ROUNDUP(N_IN*N_OUT, RF);

    float acc[N_OUT];
    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < N_OUT; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

    // core functionality
    int rufactor = RF;
    // a tmp mult for each reuse loop iteration
    float mult[multiplier_limit];

    const int ADD_LAT = DIV_ROUNDUP(multiplier_limit,N_OUT);
    //printf("rufactor = %i, add latency = %i, multiplier limit = %i \n", rufactor, ADD_LAT, multiplier_limit);
    ReuseLoop: for (int ir = 0; ir < rufactor; ir++){

        //printf("on this clock tick: %i \n", ir);
        MultLoop:
        for (int im = 0; im < multiplier_limit; im++){
            // int w_index   = ir + rufactor * im;
            int w_index   = ir * multiplier_limit + im;
            int in_index  = w_index / N_OUT;
            int out_index = w_index % N_OUT;
            if (w_index >= N_IN*N_OUT) continue; // check out of bounds
            mult[im] = data[in_index] * weights[w_index];
            // acc[out_index] += mult[im];
            // acc[out_index] += data[in_index] * weights[w_index];
            //printf("m++ ir = %i, im = %i, w_index = %i, in_index = %i, out_index = %i \n", ir, im, w_index, in_index, out_index);
        }
        // AccumLoop:
        // for (int im = 0; im < multiplier_limit; im++){
        //     int w_index   = ir + rufactor * im;
        //     int out_index = w_index % N_OUT;
        //     if (w_index >= N_IN*N_OUT) continue; // check out of bounds
        //     acc[out_index] += mult[im];
        // }

        // special loop for accumulation
        float acc_lat[N_OUT][ADD_LAT];

        AddLatencyInit:
        for (int ii = 0; ii < N_OUT; ii++){
            for (int ij= 0; ij < ADD_LAT; ij++){
                acc_lat[ii][ij] = 0;
            }
        }

        AccumLoop:
        // for (int im = 0; im < multiplier_limit; im += ADD_LAT){
        //     for (int il = 0; il < ADD_LAT; il++){
        //         // int w_index   = ir + rufactor * (im+il);
        //         int w_index   = ir * multiplier_limit + (im+il);
        //         int out_index = w_index % N_OUT;
        //         if (w_index >= N_IN*N_OUT) continue; // check out of bounds
        //         acc_lat[out_index][il] += mult[im+il];
        //         // printf("im = %i; il = %i; w_index = %i; out_index = %i \n", im, il, w_index, out_index);
        //     }
        // }
        for (int io = 0; io < N_OUT; io++){
            for (int ia = 0; ia < ADD_LAT; ia++){
                int w_index_acc    = ir * multiplier_limit + (io*ADD_LAT + ia);
                int mult_index_acc = (io*ADD_LAT + ia);
                int out_index_acc  = w_index_acc % N_OUT;

                if (mult_index_acc >= multiplier_limit) continue;
                if (w_index_acc >= N_IN*N_OUT) continue; // check out of bounds

                acc_lat[out_index_acc][ia] += mult[mult_index_acc];
                //printf("a++ ir = %i, io = %i, ia = %i, w_index = %i, out_index = %i, mult_index = %i \n", ir, io, ia, w_index_acc, out_index_acc, mult_index_acc);
            }
        }
        // printf("\n");

        FullAccum:
        for (int ii = 0; ii < N_OUT; ii++){
            for (int ij= 0; ij < ADD_LAT; ij++){
                #pragma HLS UNROLL
                acc[ii] += acc_lat[ii][ij];
            }
        }

    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < N_OUT; ires++){
        res[ires] = (float) (acc[ires]);
    }

}

void dense_v6(
    float data[N_IN],
    float res[N_OUT],
    float weights[N_IN*N_OUT],
    float biases[N_OUT]) {

    const int rufactor = RF;
    const int multfactor = MIN(N_IN,RF);
    const int totals_multipliers = N_IN*N_OUT;
    const int multiplier_limit = DIV_ROUNDUP(N_IN*N_OUT, multfactor);
    const int block_factor = DIV_ROUNDUP(N_IN*N_OUT, RF);
    const int multscale = multiplier_limit/N_OUT;
    const int nin = N_IN;
    const int nout = N_OUT;

    printf("INFO:===============================================================================\n");
    printf("n_in = %d\n", N_IN);
    printf("n_out = %d\n", N_OUT);
    printf("n_in * n_out = %d\n", N_IN * N_OUT);
    printf("RF = %d\n", RF);
    printf("block_factor = %d\n", block_factor);
    printf("multiplier_limit = %d\n", multiplier_limit);
    printf("multfactor = %d\n", multfactor);
    printf("multscale = %d\n", multscale);
    printf("INFO:===============================================================================\n");

    //assert((multiplier_limit % nin == 0) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");

    float acc[N_OUT];

    InitAccum:
    for(int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

    int in_index = 0;
    //int step_in = rufactor % nin;
    int step_in = rufactor;
    int step_loop = 0;

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++){
        //printf("--- reuse --- ir %d -----------------------------\n", ir);
        float tmpmult[block_factor];

        //printf("MultLoop:\n");
        MultLoop:
        for (int im = 0; im < block_factor; im++){
            int w_index    = ir + rufactor * im;
            //int  in_index  = w_index % nin;
            if (w_index >= N_IN*N_OUT) continue; // check out of bounds
            //printf("ir %d im %d  data[ %d ], weights[ %d ]\n", ir, im, step_loop + in_index, w_index);
            //tmpmult[im] = data[in_index] * weights[w_index];
            assert(step_loop + in_index < N_IN);
            tmpmult[im] = data[step_loop + in_index] * weights[w_index];
            in_index = in_index + step_in;
            if (in_index >= nin) {
                in_index = 0;
            }
        }
        //printf("\n");

        if (step_loop + in_index + 1 >= nin) {
        //if (step_loop + 1 >= nin) {
            step_loop = 0;
        } else {
            step_loop ++;
        }

        float mult[multiplier_limit];
        ResetMult:
        for(int imult = 0; imult < multiplier_limit; imult++) {
            mult[imult] = 0;
        }

        //printf("AccumLoop1:\n");
        AccumLoop1:
        for (int im = 0; im < block_factor; im++){
            int w_index    = ir + rufactor * im;
            int  out_index = w_index / multfactor;
            //printf("  w_index = %d, multfactor = %d\n", w_index, multfactor);

            if (out_index >= multiplier_limit) continue; // check out of bounds
            //printf("ir %d im %d  mult[ %d ], weights[ %d ]\n", ir, im, out_index, w_index);
            mult[out_index] += tmpmult[im];
            //mult[im] += tmpmult[im]; // works if RF < n_in
        }
        //printf("AccumLoop2:\n");
        AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++){
            int out_index = im/multscale;//w_index  % N_OUT;//w_index % N_OUT;//im/multscale;
            //printf("  acc[ %d ] += mult[ %d ]\n", out_index, im);
            //int out_index = w_index  % N_OUT;
            acc[out_index] += mult[im];
        }
    }

    Result:
    for(int ires = 0; ires < N_OUT; ires++){
        res[ires] = (float) (acc[ires]);
    }
}

void transpose(float arr[N_IN*N_OUT], float arr_t[N_IN*N_OUT]) {
    for(int i = 0; i < N_IN; i++) {
        for(int j = 0; j < N_OUT; j++) {
            arr_t[j * N_IN + i] = arr[i * N_OUT + j];
        }
    }
}

void print_res(char *name, float res[N_OUT]){
    printf("%s = ", name);
    for(int i = 0; i < N_OUT; i++) {
        printf("%3.0f ", res[i]);
    }
    printf("\n");
}


int main() {
    //float data[N_IN] = {70, 20, 15, 46, 28, 71, 54, 96, 94, 98, 61, 89, 16, 55, 38, 79};
    float res[N_OUT] = {0};
    float res6[N_OUT] = {0};
    //float w[N_IN*N_OUT] = {42, 72, 89, 66, 80, 6, 92, 34, 29, 9, 98, 18, 54, 73, 63, 34, 26, 95, 1, 60, 51, 52, 47, 42, 61, 32, 26, 99, 2, 38, 43, 59, 1, 95, 27, 19, 9, 61, 95, 33, 80, 84, 88, 74, 7, 100, 84, 32, 100, 57, 73, 74, 46, 6, 83, 7, 43, 97, 25, 71, 65, 30, 23, 55, 84, 29, 96, 3, 86, 58, 97, 84, 21, 51, 56, 77, 11, 81, 45, 38, 30, 22, 1, 59, 98, 89, 48, 97, 97, 42, 99, 57, 65, 81, 14, 62, 42, 27, 70, 26, 71, 47, 100, 51, 74, 40, 96, 8, 73, 81, 83, 55, 88, 3, 36, 23, 92, 39, 20, 49, 37, 97, 97, 97, 63, 39, 65, 7};
    //float b[N_OUT] = {71, 67, 77, 21, 4, 91, 1, 64};
    //float b[N_OUT] = {0};
    float data[N_IN];
    float w[N_IN*N_OUT];
    float b[N_OUT];
    float w_t[N_IN*N_OUT];

    srand(12345);
    for (int i = 0; i < N_IN; i++) {
        data[i] = (float) (rand() % 99);
    }
    for (int i = 0; i < N_IN * N_OUT; i++) {
        w[i] = (float) (rand() % 99);
    }

    transpose(w, w_t);

    dense(data, res, w, b);
    //print_res("master", res);
    //dense_v4(data, res, w, b);
    //print_res("v4    ", res);
    dense_v6(data, res6, w_t, b);
    //print_res("v6    ", res);
    int results_match = 1;
    for (int i = 0; i < N_OUT; i++) {
        if (res[i] != res6[i]) {
            results_match = 0;
            break;
        }
    }
    if (results_match) {
        printf("\n\nResults match!\n");
    } else {
        printf("\n\nResults DON'T match!\n");
    }

    return 0;
}
