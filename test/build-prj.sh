#!/bin/bash

basedir=hls_prj
hlsdir=/opt/Xilinx
hlstool=Vivado
hlsver=2020.1
hlscommand=vivado_hls
parallel=1

csim=0
synth=0
cosim=0
validation=0
vsynth=0
export=0
reset=0
fifo_opt=0

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Builds Vivado/Vitis HLS projects found in the current directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory of projects to build. Defaults to 'hls_prj'."
   echo "   -i DIR"
   echo "      Base directory of Vivado/Vitis installation. Defaults to '/opt/Xilinx'."
   echo "   -v VERSION"
   echo "      Vivado/Vitis HLS version to use. Defaults to '2020.1'."
   echo "   -p N"
   echo "      Run with N parallel tasks. Defaults to 1."
   echo "   -c"
   echo "      Run C simulation."
   echo "   -s"
   echo "      Run C/RTL synthesis."
   echo "   -r"
   echo "      Run C/RTL cosimulation."
   echo "   -t"
   echo "      Run C/RTL validation."
   echo "   -l"
   echo "      Run logic synthesis."
   echo "   -e"
   echo "      Export IP."
   echo "   -n"
   echo "      Create new project (reset any existing)."
   echo "   -a"
   echo "      Use Vitis HLS instead of Vivado HLS."
   echo "   -h"
   echo "      Prints this help message."
}

function run_hls {
   hlscommand=$1
   dir=$2
   reset=$3
   csim=$4
   synth=$5
   cosim=$6
   validation=$7
   vsynth=$8
   export=$9
   fifo_opt=${10}
   echo "Building project in ${dir} with options: reset=${reset} csim=${csim} synth=${synth} cosim=${cosim} validation=${validation} vsynth=${vsynth} export=${export} fifo_opt=${fifo_opt}"
   cd ${dir}
   cat > build_opt.tcl << EOF
array set opt {
    reset      ${reset}
    csim       ${csim}
    synth      ${synth}
    cosim      ${cosim}
    validation ${validation}
    export     ${export}
    vsynth     ${vsynth}
    fifo_opt   ${fifo_opt}
}
EOF
   cmd="\"${hlscommand}\" -f build_prj.tcl &> build_prj.log"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch BUILD_FAILED
   fi
   cd ..
   return ${failed}
}

function check_status {
   dir=$1
   cd ${dir}
   if [ -f BUILD_FAILED ]; then
      echo ""
      echo "Building project ${dir} failed. Log:"
      cat build_prj.log
      echo ""
      failed=1
   fi
   cd ..
}

while getopts ":d:i:v:p:csrtlenah" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   i) hlsdir=$OPTARG
      ;;
   v) hlsver=$OPTARG
      ;;
   p) parallel=$OPTARG
      ;;
   c) csim=1
      ;;
   s) synth=1
      ;;
   r) cosim=1
      ;;
   t) validation=1
      ;;
   l) vsynth=1
      ;;
   e) export=1
      ;;
   n) reset=1
      ;;
   a) hlstool='Vitis'
      hlscommand='vitis_hls'
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

if [ ! -d "${basedir}" ]; then
   echo "Specified directory '${basedir}' does not exist."
   exit 1
fi

#rundir=`pwd`

cd "${basedir}"

# Use .tar.gz archives to create separate project directories
for archive in *.tar.gz ; do
   filename="${archive%%.*}"
   dir="${filename}-${hlstool}-${hlsver}"
   tarpath=`tar -tf "${archive}" | grep -m1 "${filename}"`
   slashes="${tarpath//[^\/]}"
   mkdir -p "${dir}" && tar -xzf "${archive}" -C "${dir}" --strip-components ${#slashes}
done

source ${hlsdir}/${hlstool}/${hlsver}/settings64.sh

if [ "${parallel}" -gt 1 ]; then
   # Run in parallel
   (
   for dir in *-${hlstool}-${hlsver}/ ; do
      ((n=n%parallel)); ((n++==0)) && wait
      run_hls "${hlscommand}" "${dir}" "${reset}" "${csim}" "${synth}" "${cosim}" "${validation}" "${vsynth}" "${export}" "${fifo_opt}" &
   done
   wait
   )
else
   # Run sequentially
   for dir in *-${hlstool}-${hlsver}/ ; do
      run_hls "${hlscommand}" "${dir}" "${reset}" "${csim}" "${synth}" "${cosim}" "${validation}" "${vsynth}" "${export}" "${fifo_opt}"
   done
fi

# Check for build errors
for dir in *-${hlstool}-${hlsver}/ ; do
   check_status "${dir}"
done

#cd "${rundir}"

exit ${failed}
