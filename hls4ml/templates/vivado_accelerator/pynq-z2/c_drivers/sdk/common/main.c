/**
 *
 * Set Heap Size in ldscript.ld to 0x1000000 (16MB)
 *
 */

#include "xmyproject_axi.h"  /* TODO: design-dependent name */
#include "stdio.h"       /* PRINTF */
#include "unistd.h"      /* sleep */
#include "stdlib.h"
#include "malloc.h"
#include "assert.h"
#include "xil_io.h"      /* peripheral read/write wrappers */
#include "xtime_l.h"     /* to measure performance of the system */
#include "platform.h"    /* platform init/cleanup functions */
#include "xil_cache.h"   /* enable/disable caches etc */
#include "xil_printf.h"  /* UART debug print functions */
#include "xparameters.h" /* peripherals base addresses */

#include "data.h"

//#define __DEBUG__

#define MAX_PRINT_ELEMENTS (16)

#define PRINTF printf

const unsigned INPUT_N_ELEMENTS = N_SAMPLES * N_X_INPUTS;
const unsigned OUTPUT_N_ELEMENTS = N_SAMPLES * N_Y_OUTPUTS;

#if 1
/* Accelerator verification */
#define REFERENCE_OUTPUTS data_y_hls_outputs
#else
/* Accelerator validation */
#define REFERENCE_OUTPUTS data_y_outputs
//#define REFERENCE_OUTPUTS data_y_keras_outputs
#endif

unsigned get_max(float *data, unsigned n_elements) {
	float max_value = 0.0;
	unsigned max_index = 0;
	for (unsigned i = 0; i < n_elements; i++)
		if (data[i] >= max_value) {
			max_index = i;
			max_value = data[i];
		}
	return max_index;
}

float *inputs_mem = NULL;
float *outputs_mem = NULL;
float *reference_mem = NULL;

/* Accelerator configuration */
XMyproject_axi accelerator; /* TODO: design-dependent name */
XMyproject_axi_Config *accelerator_cfg; /* TODO: design-dependent name */

/* Accelerator initialization routine */
void init_accelerators() {
    PRINTF("INFO: Initializing accelerator\r\n");
    accelerator_cfg = XMyproject_axi_LookupConfig(XPAR_MYPROJECT_AXI_0_DEVICE_ID); /* TODO: design-dependent name */
    if (accelerator_cfg) {
        int status  = XMyproject_axi_CfgInitialize(&accelerator, accelerator_cfg); /* TODO: design-dependent name */
        if (status != XST_SUCCESS) {
            PRINTF("ERROR: Initializing accelerator\r\n");
        }
    }
}

/* Reference implementation of the accelerator in software */
int sw_reference_implementation(float *sw_inputs_mem, float *sw_outputs_mem, unsigned n_samples, unsigned n_X_inputs, unsigned n_y_ouputs) {
#ifdef __DEBUG__
	PRINTF("INFO: Reference outputs are pre-compiled. It would be nice to run a software model here.\r\n");
#endif
    /* See data.h for inputs and outputs */
    for (unsigned i = 0; i < n_samples * n_y_ouputs; i++) {
    	sw_outputs_mem[i] = REFERENCE_OUTPUTS[i];
    }
    return 0;
}

/* Profiling function */
double get_elapsed_time(XTime start, XTime stop) {
    return 1.0 * (stop - start) / (COUNTS_PER_SECOND);
}

/* Dump data to the console */
void dump_data(const char* label, float* data, unsigned n_samples, unsigned feature_count) {
	PRINTF("INFO:   %s[%u][%u]:\r\n", label, n_samples, feature_count);
    /* Print at most MAX_PRINT_ELEMENTS */
    for (unsigned i = 0; i < n_samples && i < MAX_PRINT_ELEMENTS; i++) {
    	PRINTF("INFO:     [%u] ", i);
        for (unsigned j = 0; j < feature_count; j++) {
        	unsigned index = i * feature_count + j;
        	PRINTF("%f ", data[index]);
        }
        PRINTF("\r\n");
    }
}

/* The top of the hill :-) */
int main(int argc, char** argv) {

    XTime start, stop;
    double calibration_time;
    double sw_elapsed = 0;
    double hw_elapsed = 0;
    double cache_elapsed = 0;
    unsigned hw_errors;

    char __attribute__ ((unused)) dummy; /* dummy input */

    /* Initialize platform (uart and caches) */
    init_platform();

    PRINTF("\r\n");
    PRINTF("INFO: ==================================================\r\n");
    PRINTF("INFO: XMyproject_axi (w/ polling)\r\n"); /* TODO: design-dependent name */
    PRINTF("INFO: ==================================================\r\n");

    init_accelerators();

    inputs_mem = malloc(INPUT_N_ELEMENTS * sizeof(float));
    outputs_mem = malloc(OUTPUT_N_ELEMENTS * sizeof(float));
    reference_mem = malloc(OUTPUT_N_ELEMENTS * sizeof(float));

    /* Calibration */
    XTime_GetTime(&start);
    sleep(1);
    XTime_GetTime(&stop);
    calibration_time = get_elapsed_time(start, stop);
    PRINTF("INFO: Time calibration for one second (%lf sec)\r\n", calibration_time);

    /* Initialize memory */
    PRINTF("INFO: Initialize memory\r\n");
    PRINTF("INFO:   - Samples count: %u\r\n", N_SAMPLES); /* Same as dst_SAMPLE_COUNT */
    PRINTF("INFO:   - Inputs count: %u\r\n", N_X_INPUTS);
    PRINTF("INFO:   - Outputs count: %u\r\n", N_Y_OUTPUTS);
    PRINTF("INFO:   - Data size: %u B\r\n", sizeof(float));
    PRINTF("INFO:   - Total input size: %u B, %.2f KB, %.2f MB\r\n", N_X_INPUTS * N_SAMPLES * sizeof(float), (N_X_INPUTS * N_SAMPLES * sizeof(float)) / (float)1024, (N_X_INPUTS * N_SAMPLES * sizeof(float)) / (float)(1024*1024));
    PRINTF("INFO:   - Total output size: %u B, %.2f KB, %.2f MB\r\n", N_Y_OUTPUTS * N_SAMPLES * sizeof(float), (N_Y_OUTPUTS * N_SAMPLES * sizeof(float)) / (float)1024, (N_Y_OUTPUTS * N_SAMPLES * sizeof(float)) / (float)(1024*1024));

    // Set Heap Size in ldscript.ld to 0x1000000 (16MB)
    //malloc_stats();

    for (int i = 0; i < INPUT_N_ELEMENTS; i++) {
        inputs_mem[i] = data_X_inputs[i];
    }
    for (int i = 0; i < OUTPUT_N_ELEMENTS; i++) {
        outputs_mem[i] = 0x0;
    }

    /* ****** SW REFERENCE ****** */
    PRINTF("INFO: ==================================================\r\n");
    PRINTF("INFO: Start SW reference implementation\r\n");
    XTime_GetTime(&start);
    sw_reference_implementation(inputs_mem, reference_mem, N_SAMPLES, N_X_INPUTS, N_Y_OUTPUTS);
    XTime_GetTime(&stop);
    sw_elapsed = get_elapsed_time(start, stop);
    PRINTF("INFO: ==================================================\r\n");
    PRINTF("INFO: Press any key to start:\r\n");
    dummy = inbyte();
    //PRINTF("INFO:");

    /* ****** HW ACCELERATOR ****** */
    PRINTF("INFO: Start HW accelerator\r\n");

    XTime_GetTime(&start);
    Xil_DCacheFlushRange((UINTPTR)inputs_mem, INPUT_N_ELEMENTS * sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)outputs_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    Xil_DCacheFlushRange((UINTPTR)reference_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    XTime_GetTime(&stop);
    cache_elapsed = get_elapsed_time(start, stop);

    for (unsigned j = 0; j < N_SAMPLES; j++) {
    	float *inputs_mem_i = inputs_mem + j * N_X_INPUTS;
    	float *outputs_mem_i = outputs_mem + j * N_Y_OUTPUTS;

    	/* Configure the accelerator */
    	XTime_GetTime(&start);
        XMyproject_axi_Set_in_r(&accelerator, (unsigned)inputs_mem_i); /* TODO: design-dependent name */
    	XMyproject_axi_Set_out_r(&accelerator, (unsigned)outputs_mem_i); /* TODO: design-dependent name */

    	XMyproject_axi_Start(&accelerator); /* TODO: design-dependent name */

    	/* Polling */
    	while (!XMyproject_axi_IsDone(&accelerator)); /* TODO: design-dependent name */

    	/* Get error status */
    	//hw_flags = XMyproject_axi_Get_return(&accelerator); /* TODO: design-dependent name */
    	XTime_GetTime(&stop);
    	hw_elapsed += get_elapsed_time(start, stop);
    }

    XTime_GetTime(&start);
    Xil_DCacheFlushRange((UINTPTR)outputs_mem, OUTPUT_N_ELEMENTS * sizeof(float));
    XTime_GetTime(&stop);
    cache_elapsed += get_elapsed_time(start, stop);

    PRINTF("INFO: HW accelerator done!\r\n");

    /* ****** VALIDATION ****** */
    PRINTF("INFO: ================== Verification ==================\r\n");
#ifdef __DEBUG__
    PRINTF("INFO: Dump data\r\n");
    dump_data("inputs_mem", inputs_mem, N_SAMPLES, N_X_INPUTS);
    dump_data("outputs_mem", outputs_mem, N_SAMPLES, N_Y_OUTPUTS);
    dump_data("reference_mem", reference_mem, N_SAMPLES, N_Y_OUTPUTS);
#endif

#ifdef __DEBUG__
    PRINTF("INFO: SW execution time: %f sec\r\n", sw_elapsed);
#endif
    PRINTF("INFO: HW-acceleration exec. time (%d inferences):\r\n", N_SAMPLES);
    PRINTF("INFO:   - total %f sec\r\n", hw_elapsed);
    PRINTF("INFO:   - per-inference %.12f sec (%f ns)\r\n", hw_elapsed / (N_SAMPLES), (hw_elapsed*1000.0) / (N_SAMPLES));
    PRINTF("INFO: Cache flush time: %f sec\r\n", cache_elapsed);
#ifdef __DEBUG__
    PRINTF("INFO: HW/SW speedup (the software is fake so this does not count...): %.2f X\r\n", (sw_elapsed >= (hw_elapsed+cache_elapsed))?(sw_elapsed/(hw_elapsed+cache_elapsed)):-((hw_elapsed+cache_elapsed)/sw_elapsed));
#endif

    hw_errors = 0;
#if 1
    /* Accelerator verification */
    for (int i = 0; i < OUTPUT_N_ELEMENTS; i++) {
        if (outputs_mem[i] != reference_mem[i]) {
            PRINTF("ERROR: [%d]: Accelerator HW %f != SW %f\r\n", i, outputs_mem[i], reference_mem[i]);
            hw_errors++;
        }
    }
    PRINTF("INFO: Total errors = %d (out of %d elements)\r\n", hw_errors, OUTPUT_N_ELEMENTS);
    if (hw_errors > 0)
        PRINTF("INFO: Verification: FAIL\r\n");
    else
        PRINTF("INFO: Verification: PASS!\r\n");
#else
    /* Accelerator validation */
    for (unsigned s = 0; s < N_SAMPLES; s++) {
    	unsigned ref_digit = get_max(reference_mem + s * N_Y_OUTPUTS, N_Y_OUTPUTS);
    	unsigned hw_digit = get_max(outputs_mem + s * N_Y_OUTPUTS, N_Y_OUTPUTS);
    	if (hw_digit != ref_digit) {
#ifdef __DEBUG__
    		PRINTF("ERROR: [%d]: Accelerator HW %u != SW %u\r\n", s, hw_digit, ref_digit);
#endif
    	    hw_errors++;
    	}
    }
    float error_rate = (hw_errors / (float)(N_SAMPLES)) * 100.0;
    float accuracy = 100 - ((hw_errors / (float)(N_SAMPLES)) * 100.0);
    PRINTF("INFO: Total errors = %d (out of %d digits)\r\n", hw_errors, N_SAMPLES);
    PRINTF("INFO: Error rate = %.2f %%\r\n", error_rate);
    PRINTF("INFO: Accuracy = %.2f %%\r\n", accuracy);
#endif
    PRINTF("INFO: ==================================================\r\n");

    cleanup_platform();

    return 0;
}


