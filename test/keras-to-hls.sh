#!/bin/bash

pycmd=python
xilinxpart="xc7vx690tffg1927-2"
clock=5
io=io_parallel
rf=1
type="ap_fixed<18,8>"
basedir=vivado_prj

sanitizer="[^A-Za-z0-9._]"

function print_usage {
   echo "Usage: `basename $0` [OPTION] MODEL[:H5FILE]..."
   echo ""
   echo "MODEL is the name of the model json file without extension. Optionally"
   echo "a H5 file with weights can be provided using the MODEL:H5FILE synthax."
   echo "By default, it is assumed that weights are stored in MODEL_weights.h5."
   echo "Multiple models can be specified."
   echo ""
   echo "Options are:"
   echo "   -p 2|3"
   echo "      Python version to use (2 or 3). If not specified uses default"
   echo "      'python' interpreter."
   echo "   -x DEVICE"
   echo "      Xilinx device part number. Defaults to 'xc7vx690tffg1927-2'."
   echo "   -c CLOCK"
   echo "      Clock period to use. Defaults to 5."
   echo "   -s"
   echo "      Use serial I/O. If not specified uses parallel I/O."
   echo "   -r FACTOR"
   echo "      Reuse factor. Defaults to 1."
   echo "   -t TYPE"
   echo "      Default precision. Defaults to 'ap_fixed<18,8>'."
   echo "   -d DIR"
   echo "      Output directory."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":p:x:c:sr:t:d:h" opt; do
   case "$opt" in
   p) pycmd=${pycmd}$OPTARG
      ;;
   x) xilinxpart=$OPTARG
      ;;
   c) clock=$OPTARG
      ;;
   s) io=io_serial
      ;;
   r) rf=$OPTARG
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
   name=${model}
   h5=${name}"_weights"
   IFS=":" read -ra model_h5_pair <<< "${model}" # If models are provided in "json:h5" format
   if [[ ${#model_h5_pair[@]} -eq 2 ]]; then
      name="${model_h5_pair[0]}"
      h5="${model_h5_pair[1]}"
   fi

   echo "Creating config file for model '${model}'"
   base=`echo "${h5}" | sed -e 's/\(_weights\)*$//g'`
   file="${basedir}/${base}-${pycmd}.yml"

   # This scheme assumes base output directory is one level deep 
   echo "KerasJson: ../../keras-to-hls/example-keras-model-files/${name}.json" > ${file}
   echo "KerasH5:   ../../keras-to-hls/example-keras-model-files/${h5}.h5" >> ${file}
   echo "OutputDir: ${base}-${pycmd}-${xilinxpart//${sanitizer}/_}-c${clock}-${io}-rf${rf}-${type//${sanitizer}/_}" >> ${file}
   echo "ProjectName: myproject" >> ${file}
   echo "XilinxPart: ${xilinxpart}" >> ${file}
   echo "ClockPeriod: ${clock}" >> ${file}
   echo "" >> ${file}
   echo "IOType: ${io}" >> ${file}
   echo "ReuseFactor: ${rf}" >> ${file}
   echo "DefaultPrecision: ${type} " >> ${file}

   ${pycmd} ../keras-to-hls/keras-to-hls.py -c ${file} || exit 1
   rm ${file}
   echo ""
done
