#!/bin/bash

pycmd=python
part="xc7vx690tffg1927-2"
clock=5
io=io_parallel
rf=1
strategy="Latency"
type="ap_fixed<16,6>"
basedir=vivado_prj

sanitizer="[^A-Za-z0-9._]"

function print_usage {
   echo "Usage: `basename $0` [OPTION] MODEL..."
   echo ""
   echo "MODEL is the name of the model onnx file without extension. Multiple"
   echo "models can be specified."
   echo ""
   echo "Options are:"
   echo "   -x DEVICE"
   echo "      Xilinx device part number. Defaults to 'xc7vx690tffg1927-2'."
   echo "   -c CLOCK"
   echo "      Clock period to use. Defaults to 5."
   echo "   -s"
   echo "      Use streaming I/O. If not specified uses parallel I/O."
   echo "   -r FACTOR"
   echo "      Reuse factor. Defaults to 1."
   echo "   -g STRATEGY"
   echo "      Strategy. 'Latency' or 'Resource'."
   echo "   -t TYPE"
   echo "      Default precision. Defaults to 'ap_fixed<16,6>'."
   echo "   -d DIR"
   echo "      Output directory."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":x:c:sr:g:t:d:h" opt; do
   case "$opt" in
   x) part=$OPTARG
      ;;
   c) clock=$OPTARG
      ;;
   s) io=io_stream
      ;;
   r) rf=$OPTARG
      ;;
   g) strategy=$OPTARG
      ;;
   t) type=$OPTARG
      ;;
   d) basedir=$OPTARG
      ;;
   h)
      print_usage
      exit
      ;;
   :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
   esac
done

shift $((OPTIND-1))

models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then
   echo "No models specified."
   exit 1
fi

mkdir -p "${basedir}"

for model in "${models[@]}"
do
   echo "Creating config file for model '${model}'"
   base=${model%.*}
   file="${basedir}/${base}.yml"

   echo "OnnxModel: ../example-models/onnx/${model}.onnx" > ${file}
   echo "OutputDir: ${basedir}/${base}-${part//${sanitizer}/_}-c${clock}-${io}-rf${rf}-${type//${sanitizer}/_}-${strategy}" >> ${file}
   echo "ProjectName: myproject" >> ${file}
   echo "Part: ${part}" >> ${file}
   echo "ClockPeriod: ${clock}" >> ${file}
   echo "" >> ${file}
   echo "IOType: ${io}" >> ${file}
   echo "HLSConfig:" >> ${file}
   echo "  Model:" >> ${file}
   echo "    ReuseFactor: ${rf}" >> ${file}
   echo "    Precision: ${type} " >> ${file}
   echo "    Strategy: ${strategy} " >> ${file}

   ${pycmd} ../scripts/hls4ml convert -c ${file} || exit 1
   rm ${file}
   echo ""
done
