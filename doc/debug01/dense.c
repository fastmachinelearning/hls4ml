#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N_IN 16
#define N_OUT 32

#define RF 8

#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

void transpose(float *data_in, float *data_out, int n_in, int n_out) {
    int i, j;
    for (i = 0; i < n_in; i++) {
        for (j = 0; j < n_out; j++) {
            int index_in = i * n_out + j;
            int index_out = j * n_in + i;
            data_out[index_out] = data_in[index_in];
        }
    }
}

int diff(float *data1, float *data2, int size) {
    int i;
    for (i = 0; i < size; i++)
        if (data1[i] != data2[i]) return 1;
    return 0;
}


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
Product1:
    for(int ii = 0; ii < N_IN; ii++) {
        cache = data[ii];
Product2:
        for(int jj = 0; jj < N_OUT; jj++) {
            int index = ii*N_OUT+jj;
            mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
ResetAccum:
    for(int iacc = 0; iacc < N_OUT; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

    // Accumulate multiplication result
Accum1:
    for(int ii = 0; ii < N_IN; ii++) {
Accum2:
        for(int jj = 0; jj < N_OUT; jj++) {
            int index = ii*N_OUT+jj;
            acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
Result:
    for(int ires = 0; ires < N_OUT; ires++){
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
ResetAccum:
    for(int iacc = 0; iacc < N_OUT; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

    // core functionality
    int rufactor = RF;
    // a tmp mult for each reuse loop iteration
    float mult[multiplier_limit];

    const int ADD_LAT = DIV_ROUNDUP(multiplier_limit,N_OUT);
    //printf("rufactor = %i, add latency = %i, multiplier limit = %i \n", rufactor, ADD_LAT, multiplier_limit);
ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++){

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

    //printf("block_factor = %d\n", block_factor);
    //printf("multiplier_limit = %d\n", multiplier_limit);

    assert((multiplier_limit % nin == 0) && "The current Reuse Factor is not allowed");

    float acc[N_OUT];

InitAccum:
    for(int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (float) biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++){
        float tmpmult[block_factor];

MultLoop:
        for (int im = 0; im < block_factor; im++){
            int w_index    = ir + rufactor * im;
            int  in_index  = w_index % nin;
            //int  out_index = w_index / nin;
            if (w_index >= N_IN*N_OUT) continue; // check out of bounds
            assert(im < block_factor);
            assert(in_index < N_IN);
            assert(w_index < N_IN*N_OUT);
            tmpmult[im] = data[in_index] * weights[w_index];
        }

        float mult[multiplier_limit];
ResetMult:
        for(int imult = 0; imult < multiplier_limit; imult++) {
            mult[imult] = 0;
        }

AccumLoop1:
        for (int im = 0; im < block_factor; im++){
            int w_index    = ir + rufactor * im;
            int  out_index = w_index / multfactor;
            if (out_index >= multiplier_limit) continue; // check out of bounds
            assert(out_index < multiplier_limit);
            assert(im < block_factor);
            mult[out_index] += tmpmult[im];
        }

AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++){
            //int w_index   = ir + rufactor * im;
            //if (w_index >= N_IN*N_OUT) std::cout << " ---> " << N_IN*N_OUT << " -- " << im << " -- " << w_index << " -- " << block_factor << std::endl;
            int out_index = im / multscale;//w_index  % N_OUT;//w_index % N_OUT;//im/multscale;
            assert(out_index < N_OUT);
            assert(im < multiplier_limit);
            acc[out_index] += mult[im];
        }

    }

Result:
    for(int ires = 0; ires < N_OUT; ires++){
        assert(ires < N_OUT);
        res[ires] = (float) (acc[ires]);
    }
}

void print_res(char *name, float res[N_OUT]){
    printf("%s = ", name);
    for(int i = 0; i < N_OUT; i++) {
        printf("%3.0f ", res[i]);
    }
    printf("\n");
}

int main(int argc, char *argv) {
    float data[N_IN];
    float res_master[N_OUT] = {0};
    float res_v4[N_OUT] = {0};
    float res_v6[N_OUT] = {0};

    float w[N_IN*N_OUT];
    float wT[N_IN*N_OUT];

    for (int i = 0; i < N_IN; i++)
        data[i] = i + 0.5;

#ifdef ONES
    for (int i = 0; i < N_IN * N_OUT; i++)
        w[i] = 1.0;
#else
    for (int i = 0; i < N_IN * N_OUT; i++)
        w[i] = (float)i;
 #endif

#if 0
    for (int i = 0; i < N_OUT; i++)
        b[i] = i * 0.025;
#else
    float b[N_OUT] = {0};
#endif

    transpose(w, wT, N_IN, N_OUT); // <---------------------- dense_v6 works on transposed weights

    dense(data, res_master, w, b);
    print_res("INFO: master", res_master);
    dense_v4(data, res_v4, w, b);
    print_res("INFO: v4    ", res_v4);
    dense_v6(data, res_v6, wT, b);
    print_res("INFO: v6    ", res_v6);

    printf("INFO: master vs. v4: %s\n", diff(res_master, res_v4, N_OUT) ? "FAIL" : "PASS");
    printf("INFO: master vs. v6: %s\n", diff(res_master, res_v6, N_OUT) ? "FAIL" : "PASS");

    return 0;
}
