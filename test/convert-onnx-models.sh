#!/bin/bash

models_var=HLS4ML_ONNX_MODELS
models_file=
exec=echo
dir=

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Reads the model names from the ${models_var} environment variable"
   echo "or provided file name and optionally starts the conversion."
   echo ""
   echo "Options are:"
   echo "   -f FINENAME"
   echo "      File name to read models from. If not specified, reads from ${models_var}"
   echo "      environment variable."
   echo "   -x"
   echo "      Execute the commands instead of just printing them."
   echo "   -d DIR"
   echo "      Output directory passed to onnx-to-hls script."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":f:xd:h" opt; do
   case "$opt" in
   f) models_file=$OPTARG
      ;;
   x) exec=eval
      ;;
   d) dir="-d $OPTARG"
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

if [ -z ${models_file} ]; then
   if [ -z ${!models_var+x} ] ; then
      echo "No file provided and ${models_var} variable not set. Nothing to do."
      exit 1
   else
      IFS=";" read -ra model_line <<< "${!models_var}"
   fi
else
   readarray model_line < "${models_file}"
fi

for line in "${model_line[@]}"
do
   params=("" "" "" "" "" "")
   if [[ ${line} = *[![:space:]]* ]] && ! [[ "${line}" = \#* ]] ; then
      IFS=" " read -ra model_def <<< "${line}"
      for (( i=1; i<"${#model_def[@]}"; i++ ));
      do
         if [[ "${model_def[$i]}" == x:* ]] ; then params[0]="-x ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == c:* ]] ; then params[1]="-c ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == io:s ]] ; then params[2]="-s "; fi
         if [[ "${model_def[$i]}" == r:* ]] ; then params[3]="-r ${model_def[$i]:2} "; fi
         if [[ "${model_def[$i]}" == i:* ]] ; then params[4]="-t ${model_def[$i]:2} "; fi
      done
      params[5]=${model_def[0]}

      cmd="./onnx-to-hls.sh ${dir} ${params[0]}${params[1]}${params[2]}${params[3]}${params[4]}${params[5]}"

      ${exec} "${cmd}"
   fi
done

#cd "${rundir}"

exit ${failed}
