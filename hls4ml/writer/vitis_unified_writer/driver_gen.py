import os
import stat
from pathlib import Path


from .meta import VitisUnifiedWriterMeta

class VitisUnified_DriverGen:

    @classmethod
    def write_driver(self, meta, model, mg):
        filedir = os.path.dirname(os.path.abspath(__file__))
        fin     = open(os.path.join(filedir, '../../templates/vitis_unified/driver/pynq/pynq_driver.py'), 'r')
        fout    = open(f'{model.config.get_output_dir()}/export/pynq_driver.py', 'w')

        inp_gmem_t, out_gmem_t, inps, outs = meta.vitis_unified_config.get_corrected_types()

        strideInPtrAddr   = 4*3
        strideOutPtrAddr  = 4*3
        strideInSizeAddr  = 4*2
        strideOutSizeAddr = 4*2

        startInPtrAddr = 0x10
        startOutPtrAddr = startInPtrAddr   + strideInPtrAddr  * len(inps)
        startInSizeAddr = startOutPtrAddr  + strideOutPtrAddr * len(outs)
        startOutSizeAddr = startInSizeAddr + strideInSizeAddr * len(inps)

        def genHexAddrList(startAddr, stride, size, indent):
            addrs = [f"{indent}{hex(startAddr + inp_idx * stride)}" for inp_idx in range(size)]
            return addrs

        indentAmt = 3
        indentStr = indentAmt * "    " if indentAmt > 0 else ""

        for line in fin.readlines():

            if "#### hls-driver-input-dbg-name" in line:
                input_names = [ f'{indentStr}"{mg.get_io_port_name(inp, True, idx)}"' for idx, inp in enumerate(inps) ]
                line += ",\n".join(input_names) + "\n"
            if "#### hls-driver-input-ptr" in line:
                line += ",\n".join(genHexAddrList(startInPtrAddr, strideInPtrAddr, len(inps), indentStr)) + "\n"
            if "#### hls-driver-input-size" in line:
                line += ",\n".join(genHexAddrList(startInSizeAddr, strideInSizeAddr, len(inps), indentStr)) + "\n"
            if "#### hls-driver-output-dbg-name" in line:
                output_names = [ f'{indentStr}"{mg.get_io_port_name(out, False, idx)}"' for idx, out in enumerate(outs) ]
                line += ",\n".join(output_names) + "\n"
            if "#### hls-driver-output-ptr" in line:
                line += ",\n".join(genHexAddrList(startOutPtrAddr, strideOutPtrAddr, len(outs), indentStr)) + "\n"
            if "#### hls-driver-output-size" in line:
                line += ",\n".join(genHexAddrList(startOutSizeAddr, strideOutSizeAddr, len(outs), indentStr)) + "\n"
            if "<TOP_NAME>" in line:
                line = line.replace("<TOP_NAME>", mg.get_top_wrap_func_name(model))

            fout.write(line)

        fin.close()
        fout.close()