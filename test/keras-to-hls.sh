#!/bin/bash

pycmd=python
part="xcvu9p-flgb2104-2-e"
board="None"
backend="Vivado"
clock=5
io=io_parallel
rf=1
strategy="Latency"
type="ap_fixed<16,6>"
yml=""
basedir=hls_prj
precision="float"
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
   echo "   -x PART"
   echo "      FPGA device part number. Defaults to 'xcvu9p-flgb2104-2-e'."
   echo "   -b BOARD"
   echo "      Board used. Defaults to 'pynq-z2'."
   echo "   -B BACKEND"
   echo "      Backend to use for the generation of the code. Defaults to 'Vivado'."
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
   echo "   -y FILE"
   echo "      YAML config file to take HLS config from. If specified, -r, -g and -t are ignored."
   echo "   -P PYCMD"
   echo "      python command. Default is 'python'."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":x:b:B:c:sr:g:t:d:y:p:P:h" opt; do
   case "$opt" in
   x) part=$OPTARG
      ;;
   b) board=$OPTARG
      ;;
   B) backend=$OPTARG
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
   y) yml=$OPTARG
      ;;
   p) precision=$OPTARG
      ;;
   P) pycmd=$OPTARG
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
   file="${basedir}/${base}.yml"
   prjdir="${basedir}/${base}-backend${backend}-board${board//${sanitizer}/_}-${part//${sanitizer}/_}-c${clock}-${io}-rf${rf}-${type//${sanitizer}/_}-${strategy}"

   hlscfg=""
   if [ ! -z "${yml}" ]; then
      hlscfg=`sed -ne '/HLSConfig/,$p' ../example-models/config-files/${yml}`
   fi
   echo "KerasJson: ../example-models/keras/${name}.json" > ${file}
   echo "KerasH5:   ../example-models/keras/${h5}.h5" >> ${file}
   echo "OutputDir: ${prjdir}" >> ${file}
   echo "ProjectName: myproject" >> ${file}
   echo "Part: ${part}" >> ${file}
   echo "Board: ${board}" >> ${file}
   echo "Backend: ${backend}" >> ${file}
   echo "ClockPeriod: ${clock}" >> ${file}
   echo "" >> ${file}
   echo "IOType: ${io}" >> ${file}
   if [ -z "${hlscfg}" ]
   then
      echo "HLSConfig:" >> ${file}
      echo "  Model:" >> ${file}
      echo "    ReuseFactor: ${rf}" >> ${file}
      echo "    Precision: ${type} " >> ${file}
      echo "    Strategy: ${strategy} " >> ${file}
   else
      echo "${hlscfg}" >> ${file}
   fi
   # Adding VivadoAccelerator config to file
   if [ "${backend}" = "VivadoAccelerator" ];
   then
     echo "AcceleratorConfig:" >> ${file}
     echo "  Board: ${board}" >> ${file}
     echo "  Precision:" >> ${file}
     echo "    Input: ${precision}" >> ${file}
     echo "    Output: ${precision}" >> ${file}
   fi
   # Write tarball
   echo "WriterConfig:" >> ${file}
   echo "  Namespace: null" >> ${file}
   echo "  WriteWeightsTxt: true" >> ${file}
   echo "  WriteTar: true" >> ${file}

   ${pycmd} ../scripts/hls4ml convert -c ${file} || exit 1
   rm ${file}
   rm -rf "${prjdir}"
   echo ""
done
