#pragma once

#include "FpgaObj.hpp"

template <class V, class W>
class DdrFpga : public FpgaObj<V, W> {
 public:
    DdrFpga(int kernInputSize, int kernOutputSize, int numCU, int numThreads, int numEpochs)
        : FpgaObj<V, W>(kernInputSize, kernOutputSize, numCU, numThreads, numEpochs) {
    }

    void allocateHostMemory(int chan_per_port) {
        // Creating Buffer objects in Host memory
        /* ***NOTE*** When creating a Buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood, user pointer
        is used if it is properly aligned. when not aligned, runtime has no choice but to create
        its own host side Buffer. So it is recommended to use this allocator if user wishes to
        create Buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
        ensure that user buffer is used when user creates Buffer/Mem object with CL_MEM_USE_HOST_PTR */
        size_t vector_size_in_bytes = sizeof(V) * this->_kernInputSize;
        size_t vector_size_out_bytes = sizeof(W) * this->_kernOutputSize;
        for (int ib = 0; ib < this->_numThreads; ib++) {
            for (int ik = 0; ik < this->_numCU; ik++) {
                cl::Buffer buffer_in_tmp(this->context, 
                        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                        vector_size_in_bytes,
                        this->source_in.data() + ((ib*this->_numCU + ik) * this->_kernInputSize));
                cl::Buffer buffer_out_tmp(this->context,
                        CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                        vector_size_out_bytes,
                        this->source_hw_results.data() + ((ib*this->_numCU + ik) * this->_kernOutputSize));
                this->buffer_in.push_back(buffer_in_tmp);
                this->buffer_out.push_back(buffer_out_tmp);
                this->krnl_xil[ib*this->_numCU + ik].setArg(0, this->buffer_in[ib*this->_numCU + ik]);
                this->krnl_xil[ib*this->_numCU + ik].setArg(1, this->buffer_out[ib*this->_numCU + ik]);
            }
        }
    }
};