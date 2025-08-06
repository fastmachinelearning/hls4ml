#ifndef HOST_LIBS_HPP_
#define HOST_LIBS_HPP_

#include <vector>
#include "cOps.hpp"
#include "cThread.hpp"

// Coyote uses so-called vFPGAs: individual applications running in parallel on the FPGA
// Users can deploy multiple vFPGAs on the same hardware, each with its own application
// For now, the CoyoteAccelerator only supports a single vFPGA, though future extensions
// could easily allow multiple parallel instance of hls4ml models
#define DEFAULT_VFPGA_ID 0

/**
  * @brief Utility class for running inference of an hls4ml model with the Coyote accelerator backend
  *
  * This class can be used to set up and execute the inference, by allocating memory for the tensors, 
  * running the inference, and retrieving predictions. It abstracts away all the interaciton with the 
  * Coyote software library, which in turn abstracts away the interaction with the hardware.
  * This class assumes some familiarity with the Coyote software library; examples of its use
  * can be found on Github examples: https://github.com/fpgasystems/Coyote/tree/master/examples.
  *
  * NOTE: This class can be linked into a shared library and called from the Python overlay (CoyoteOverlay) or
  * it can be instantiated stand-alone in a C++ code.
  *
  * NOTE: The functions set_data, predict and get_prediction are separated, simply to be able to obtain granular
  * measurements of how long each step takes. One could easily combine them into a single function. 
  
  * NOTE: There is a  difference between XRT (VitisAccelerator backend) and Coyote: in XRT it is necessary 
  * to sync the input data from the host memory to device memory (HBM/DDR) befor running the inference. 
  * On the other hand, Coyote implements a shared virtual memory model, and the shell will automatically
  * fetch data from host memory and feed it to the model kernel, fully bypassing device memory. However,
  * we still have a function set_data that esentially copies data from one host-side array (e.g., NumPy) to
  * an array that's a member variable of this class. This is not necessary and Coyote could equally work
  * with the NumPy array, but it makes it easier to manage multiple batches. Future optimizations could fix
  * this, if desired. For more details on Coyote's memory model, refer to the paper: https://arxiv.org/abs/2504.21538 
 */
class CoyoteInference {
public:
    /**
     * @brief Constructor for CoyoteInference
     * @param batch_size Number of samples in a batch
     * @param in_size Size of the input tensor (in elements)
     * @param out_size Size of the output tensor (in elements)
     *
     * NOTE: The batch size is not a hardware/synthesis parameter, but rather a runtime parameter
     * Coyote supports asynchronous execution of request, so the software can invoke multiple 
     * inputs, as specified by the batch size, and the hardware handles the scheduling, any back-pressure etc.
     */
    CoyoteInference(unsigned int batch_size, unsigned int in_size, unsigned int out_size);

    /// Default destructor
    ~CoyoteInference();

    /**
     * @brief Utility function, clears completion counters in Coyote and resets output tensors to zero
     */
    void flush();

    /**
     * @brief Runs inference on the input tensors, specified by set_data
     */
    void predict();

    /**
     * @brief Set the input data for a specific entry of the batch
     *
     * @param x Pointer to the input data (array of floats)
     * @param i Index of the batch entry to set data for
     */
    void set_data(float *x, unsigned int i);

    /**
     * @brief Returns the i-th prediction of a batch
     *
     * @param i Index of the batch entry to get predictions for
     * @return Pointer to the output predictions (array of floats)
     */
    float* get_predictions(unsigned int i);

private:

    unsigned int batch_size, in_size, out_size;
    
    /**
     * @brief Coyote thread for inference
     * 
     * Coyote uses so called threads to interfact with th FPGA, which include
     * high-level functions for moving data, setting control registers,
     * polling on completions etc.
     */
    coyote::cThread coyote_thread;
    
    /**
     * @brief Coyote scatter-gather entries
     * 
     * Scatter-gather entries are used to specify the source and destination
     * addresses and lengths for data transfers between host memory and the FPGA.
     * In this case, they point to the input and output tensors for each batch entry.
     */
    std::vector<coyote::localSg> src_sgs, dst_sgs;
    
    /// Memory pointers for input tensors (one per batch entry)
    std::vector<float*> src_mems, dst_mems;
};

#endif
