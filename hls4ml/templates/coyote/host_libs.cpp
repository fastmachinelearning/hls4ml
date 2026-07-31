#include "host_libs.hpp"

CoyoteInference::CoyoteInference(unsigned int batch_size, unsigned int in_size, unsigned int out_size)
    : batch_size(batch_size), in_size(in_size), out_size(out_size), coyote_thread(DEFAULT_VFPGA_ID, getpid()) {
    for (unsigned int i = 0; i < batch_size; i++) {
        // Allocate memory using huge pages (HPF) for input and output tensors
        src_mems.emplace_back(
            (float *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, (uint)(in_size * sizeof(float))}));
        dst_mems.emplace_back(
            (float *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, (uint)(out_size * sizeof(float))}));
        if (!src_mems[i] || !dst_mems[i]) {
            throw std::runtime_error("Could not allocate memory; exiting...");
        }

        // Create scatter-gather entry for this input/output pair
        coyote::localSg src_sg = {.addr = src_mems[i], .len = (uint)(in_size * sizeof(float))};
        coyote::localSg dst_sg = {.addr = dst_mems[i], .len = (uint)(out_size * sizeof(float))};
        src_sgs.emplace_back(src_sg);
        dst_sgs.emplace_back(dst_sg);
    }
}

CoyoteInference::~CoyoteInference() {}

void CoyoteInference::flush() {
    // Reset output tensors to zero
    for (unsigned int i = 0; i < batch_size; i++) {
        memset(dst_mems[i], 0, out_size);
    }

    // Clear completion counters
    coyote_thread.clearCompleted();
}

void CoyoteInference::predict() {
    // Coyote uses the so-called invoke function to run operation in vFPGAs.
    // In this case, the operation is LOCAL_TRANSFER, and the flow of data is:
    // host memory (input data) => vFPGA (hls4ml model) => host memory (output data)
    for (int i = 0; i < batch_size; i++) {
        coyote_thread.invoke(coyote::CoyoteOper::LOCAL_TRANSFER, src_sgs[i], dst_sgs[i]);
    }

    // Poll on completion; each batch increments the counter by one
    while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) != batch_size) {
    }
}

void CoyoteInference::set_data(float *x, unsigned int i) {
    // Simply copy from one buffer to the other
    for (int j = 0; j < in_size; j++) {
        src_mems[i][j] = x[j];
    }
}

float *CoyoteInference::get_predictions(unsigned int i) { return dst_mems[i]; }

// C API for the CoyoteInference class; so that it can be used from Python or other languages
// Better option would be to use something like pybind11, but the implementation is simple enough for now.
extern "C" {
CoyoteInference *init_model_inference(unsigned int batch_size, unsigned int in_size, unsigned int out_size) {
    return new CoyoteInference(batch_size, in_size, out_size);
}

void free_model_inference(CoyoteInference *obj) { delete obj; }

void flush(CoyoteInference *obj) { obj->flush(); }

void predict(CoyoteInference *obj) { obj->predict(); }

void set_inference_data(CoyoteInference *obj, float *x, unsigned int i) { obj->set_data(x, i); }

float *get_inference_predictions(CoyoteInference *obj, unsigned int i) { return obj->get_predictions(i); }
}
