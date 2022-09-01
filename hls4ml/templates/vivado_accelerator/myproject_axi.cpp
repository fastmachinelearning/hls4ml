//hls-fpga-machine-learning insert include

void myproject(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        ){

    //hls-fpga-machine-learning insert interface

    //hls-fpga-machine-learning insert local vars

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }
#endif

    //hls-fpga-machine-learning insert enqueue

    //hls-fpga-machine-learning insert call

    //hls-fpga-machine-learning insert dequeue
}
