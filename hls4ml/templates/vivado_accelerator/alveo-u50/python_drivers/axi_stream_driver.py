from datetime import datetime

import numpy as np
from pynq import Overlay
from pynq import allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(self, xclbin_name, dtbo=None, download=True, ignore_version=False, device=None):

        super().__init__(xclbin_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)

    def print_dt(self, timea, timeb, N):
        dt = (timeb - timea)
        dts = dt.seconds + dt.microseconds * 10 ** -6
        rate = N / dts
        print("Classified {} samples in {} seconds ({} inferences / s)".format(N, dts, rate))
        print("Or {} us / inferences".format(1 / rate * 1e6))
        return dts, rate

    def predict(self, X, y_shape, dtype=np.float32, debug=None, profile=False, encode=None, decode=None):
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
        # TODO Improve the memory target definition (memory target should be chosen accordigly to the design)
        with allocate(shape=X.shape, dtype=dtype, target=self.HBM0) as input_buffer, \
                allocate(shape=y_shape, dtype=dtype, target=self.HBM1) as output_buffer:
            in_size = np.prod(X.shape)
            out_size = np.prod(y_shape)
            input_buffer[:] = X
            input_buffer.sync_to_device()
            if debug:
                print("Send OK")
            self.krnl_rtl_1.call(input_buffer, output_buffer, in_size, out_size)
            if debug:
                print("Kernel call OK")
            output_buffer.sync_from_device()
            if debug:
                print("Receive OK")
            result = output_buffer.copy()
            input_buffer.flush()
            output_buffer.flush()
            del input_buffer
            del output_buffer
            self.free()
        if decode is not None:
            result = decode(result)
        if profile:
            timeb = datetime.now()
            dts, rate = self.print_dt(timea, timeb, len(X))
            return result, dts, rate
        return result
