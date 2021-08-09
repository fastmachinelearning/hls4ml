from pynq import DefaultHierarchy, allocate
from datetime import datetime
import pynq.lib.dma
import numpy as np
# todo: see below
'''
ideas:
    - generate the proper driver from hls4ml (axi_s/m_axi/axi_lite) starting from a common template that will be
    filled with the proper code. For predict(), for example, can be imported another file with the correct 
    implementation of predict() function (depending on which system - axi_s, axi_m, axi_lite - the user would like to use).
'''


class NN(DefaultHierarchy):
    """
    Hierarchy driver related to the hierarchy composed by the AXI DMA module + the HLS accelerator.
    It uses the AxiLiteNN class to write the number of rows of the input matrix to the accelerator.
    """

    def __init__(self, description):
        super().__init__(description=description)

    def _print_dt(self, timea, timeb, N):
        dt = (timeb - timea)
        dts = dt.seconds + dt.microseconds * 10 ** -6
        rate = N / dts
        print("Classified {} samples in {} seconds ({} inferences / s)".format(N, dts, rate))
        return dts, rate

    def predict(self, X, y_shape, dtype=np.float32, profile=False, debug=False, encode=None, decode=None):
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - X : the input vector. Should be numpy ndarray.
        - y_shape : the shape of the output vector. Needed to the accelerator to set the TLAST bit properly and
                    for sizing the output vector shape.
        - dtype : the data type of the elements of the input/output vectors. 
                  Note: it should be set depending on the interface of the accelerator; if it uses 'float' 
                  types for the 'data' AXI-Stream field, 'np.float32' dtype is the correct one to use. 
                  Instead if it uses 'ap_fixed<A,B>', 'np.intA' is the correct one to use (note that A cannot
                  any integer value, but it can assume {..., 8, 16, 32, ...} values. Check `numpy` 
                  doc for more info).
                  In this case the encoding/decoding has to be computed by the PS. For example for 
                  'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode 
                  'float' -> 'ap_fixed<16,6>':

                  ```
                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)

                    def decode(yi):
                        return yi * 2**-10

                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)
                  ```
        - profile : boolean. Set it to `True` to print the performance of the algorithm in term of `inference/s`.
        - encode/decode: function pointers. See `dtype` section for more information.
        - return: an output array based on `np.ndarray` with a shape equal to `y_shape` and a `dtype` equal to
                  the namesake parameter.

        """
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        with allocate(shape=X.shape, dtype=dtype) as input_buffer, \
                allocate(shape=y_shape, dtype=dtype) as output_buffer:
            input_buffer[:] = X
            self.axi_dma_0.sendchannel.transfer(input_buffer)
            self.axi_dma_0.recvchannel.transfer(output_buffer)
            if debug:
                print('Data transfers OK!')
            self.axi_dma_0.sendchannel.wait()
            if debug:
                print('sendchannel.wait() completed!')
            self.axi_dma_0.recvchannel.wait()
            if debug:
                print('recvchannel.wait() completed!')
            result = output_buffer.copy()
        if decode is not None:
            result = decode(result)
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return result, dts, rate
        return result

    @staticmethod
    def checkhierarchy(description):
        if 'axi_dma_0' in description['ip']:
            return True
        return False
