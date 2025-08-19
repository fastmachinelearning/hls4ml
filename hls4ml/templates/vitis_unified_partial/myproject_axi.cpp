#include <stdint.h>
#include <hls_stream.h>
#include <iostream>



template<typename INPUT_LAYER_ARR, int SIZE>
void enqueue_atom2layer(hls::stream<dma_data_packet>& src_dma_stream, hls::stream<INPUT_LAYER_ARR>& raw_stream, bool& isLastIndicate){

    dma_data_packet dma_tmp;
    for (int i = 0; i < (SIZE/INPUT_LAYER_ARR::size); i++){
        INPUT_LAYER_ARR ctype;
        for (int j = 0; j < INPUT_LAYER_ARR::size; j++){
            src_dma_stream.read(dma_tmp);
            ctype[j] = dma_tmp.data;
            isLastIndicate = dma_tmp.last;
        }
        raw_stream.write(ctype);
    }

}

template<typename INPUT_LAYER_STREAM, typename INPUT_LAYER_ARR, int SIZE>
void enqueue_layerStream2layer(hls::stream<INPUT_LAYER_STREAM>& src_mgs_stream, hls::stream<INPUT_LAYER_ARR>& raw_stream, bool& isLastIndicate){

    INPUT_LAYER_STREAM input_layer_stream_tmp;
    for (unsigned i = 0; i < (SIZE/INPUT_LAYER_ARR::size); i++){
        INPUT_LAYER_ARR ctype;
        src_mgs_stream.read(input_layer_stream_tmp);
        ctype = input_layer_stream_tmp.data;
        isLastIndicate = input_layer_stream_tmp.last;
        raw_stream.write(ctype);
    }
    input_layer_stream_tmp.last = 0;

}


template<typename ATOMIC_TYPE, typename OUTPUT_LAYER_ARR, int SIZE>
void dequeue_layer2atom(hls::stream<dma_data_packet>& des_dma_stream, hls::stream<OUTPUT_LAYER_ARR>& raw_stream, bool& isLastIndicate){
    dma_data_packet dma_tmp;
    dma_tmp.last = 0;
    for(unsigned i = 0; i < SIZE/OUTPUT_LAYER_ARR::size; ++i){
        OUTPUT_LAYER_ARR ctype = raw_stream.read();
        for(unsigned j = 0; j < OUTPUT_LAYER_ARR::size; ++j){
                dma_tmp.data = (ATOMIC_TYPE) (ctype[j]);
                if(isLastIndicate){
                    dma_tmp.last = (((i+1)*(j+1))==SIZE);
                }
                des_dma_stream.write(dma_tmp);
        }
    }
    dma_tmp.last = 0;
}

template<typename OUTPUT_LAYER_STREAM, typename OUTPUT_LAYER_ARR, int SIZE>
void dequeue_layer2layer(hls::stream<OUTPUT_LAYER_STREAM>& des_mgs_stream, hls::stream<OUTPUT_LAYER_ARR>& raw_stream, bool& isLastIndicate){

    OUTPUT_LAYER_STREAM output_layer_stream_tmp;
    output_layer_stream_tmp.last = 0;
    for (unsigned i = 0; i < (SIZE/OUTPUT_LAYER_ARR::size); i++){
        OUTPUT_LAYER_ARR ctype = raw_stream.read();
        output_layer_stream_tmp.data = ctype;
        if(isLastIndicate){
            output_layer_stream_tmp.last = ((i+1) == (SIZE/OUTPUT_LAYER_ARR::size));
        }
        des_mgs_stream.write(output_layer_stream_tmp);
    }

}

void MY_PROJECT_TOP_FUNC(
// hls-fpga-machine-learning insert multi-io

) {

// hls-fpga-machine-learning insert interface

// hls-fpga-machine-learning insert local vars

// hls-fpga-machine-learning insert enqueue

// hls-fpga-machine-learning insert call

// hls-fpga-machine-learning insert dequeue
}
