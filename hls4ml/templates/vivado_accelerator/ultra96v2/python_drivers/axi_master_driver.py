from pynq import DefaultHierarchy, DefaultIP, allocate
from pynq import Overlay
from datetime import datetime
import pynq.lib.dma
import numpy as np


class NeuralNetworkOverlay(Overlay):
    #def __init__(self, bitfile_name, x_shape, y_shape, w2_shape, b2_shape, w5_shape, b5_shape, dtype=np.float32, dtbo=None, download=True, ignore_version=False, device=None):
    #hls-fpga-machine-learning insert init
        super().__init__(bitfile_name, dtbo=None, download=True, ignore_version=False, device=None)
        self.regin = self.myproject_axi_0.register_map.in_r.address
        self.regout = self.myproject_axi_0.register_map.out_r.address
        #
        #self.regw2 = self.myproject_axi_0.register_map.w2.address
        #self.regb2 = self.myproject_axi_0.register_map.b2.address
        #self.regw5 = self.myproject_axi_0.register_map.w5.address
        #self.regb5 = self.myproject_axi_0.register_map.b5.address
        #hls-fpga-machine-learning insert registers
        self.reglw = self.myproject_axi_0.register_map.load_weights.address
        #
        self.ctrl = self.myproject_axi_0.register_map.CTRL
        self.input_buffer = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype)
        #self.w2_buffer = allocate(shape=w2_shape, dtype=dtype)
        #self.b2_buffer = allocate(shape=b2_shape, dtype=dtype)
        #self.w5_buffer = allocate(shape=w5_shape, dtype=dtype)
        #self.b5_buffer = allocate(shape=b5_shape, dtype=dtype)
        #hls-fpga-machine-learning insert buffers
    def _print_dt(self, timea, timeb, N):
        dt = (timeb - timea)
        dts = dt.seconds + dt.microseconds * 10 ** -6
        rate = N / dts
        print("Classified {} samples in {} seconds ({} inferences / s)".format(N, dts, rate))
        return dts, rate

#    def load_weights(self, w2, b2, w5, b5, debug=False, profile=False, encode=None):
        #hls-fpga-machine-learning insert load weights
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - w*, b* : the weight and bias vectors. Should be numpy ndarray.
        - profile : boolean. Set it to `True` to print the performance of the algorithm in term of `inference/s`.
        - encode: function pointers. See `dtype` section for more information.
        """
        if profile:
            timea = datetime.now()
        if encode is not None:
            #w2 = encode(w2)
            #b2 = encode(b2)
            #w5 = encode(w5)
            #b5 = encode(b5)
            #hls-fpga-machine-learning insert encode
        #
        #self.w2_buffer[:] = w2
        #self.b2_buffer[:] = b2
        #self.w5_buffer[:] = w5
        #self.b5_buffer[:] = b5
        #hls-fpga-machine-learning insert set buffers
        #
        #self.myproject_axi_0.write(self.regw2, self.w2_buffer.physical_address)
        #self.myproject_axi_0.write(self.regb2, self.b2_buffer.physical_address)
        #self.myproject_axi_0.write(self.regw5, self.w5_buffer.physical_address)
        #self.myproject_axi_0.write(self.regb5, self.b5_buffer.physical_address)
        #hls-fpga-machine-learning insert set registers
        #
        self.myproject_axi_0.write(self.reglw, 0x1)
        #
        self.myproject_axi_0.write(self.ctrl.AP_START, 0x1)
        if debug:
            print("Config OK")
        while not self.ctrl.AP_DONE:
            if debug:
                print("Polling...")
        if debug:
            print("Done OK")
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))


    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - X : the input vector. Should be numpy ndarray.
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
        self.input_buffer[:] = X
        self.myproject_axi_0.write(self.regin, self.input_buffer.physical_address)
        self.myproject_axi_0.write(self.regout, self.output_buffer.physical_address)
        #
        self.myproject_axi_0.write(self.reglw, 0x0)
        #
        self.myproject_axi_0.write(self.ctrl.AP_START, 0x1)
        if debug:
            print("Config OK")
        while not self.ctrl.AP_DONE:
            if debug:
                print("Polling...")
        if debug:
            print("Done OK")
        # result = self.output_buffer.copy()
        if decode is not None:
            self.output_buffer = decode(self.output_buffer)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return self.output_buffer, dts, rate
        else:
            return self.output_buffer
