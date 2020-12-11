#!/bin/bash

basedir=vivado_prj
vivadodir=/opt/Xilinx
vivadover=2020.1
parallel=1

csim="csim=0"
synth="synth=0"
cosim="cosim=0"
validation="validation=0"
vsynth="vsynth=0"
export="export=0"
reset="reset=0"

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Builds Vivado HLS projects found in the current directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory of projects to build. Defaults to 'vivado_prj'."
   echo "   -i DIR"
   echo "      Base directory of Vivado installation. Defaults to '/opt/Xilinx'."
   echo "   -v VERSION"
   echo "      Vivado HLS version to use. Defaults to '2020.1'."
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
   echo "      Run Vivado (logic) synthesis."
   echo "   -e"
   echo "      Export IP."
   echo "   -n"
   echo "      Create new project (reset any existing)."
   echo "   -h"
   echo "      Prints this help message."
}

function run_vivado {
   dir=$1
   opt=$2
   echo "Building project in ${dir} with options: ${opt}"
   cd ${dir}
   cmd="vivado_hls -f build_prj.tcl \"${opt}\" &> build_prj.log"
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
      echo "Building project ${dir} (${opt}) failed. Log:"
      cat build_prj.log
      echo ""
      failed=1
   fi
   cd ..
}

while getopts ":d:i:v:p:csrtlenh" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   i) vivadodir=$OPTARG
      ;;
   v) vivadover=$OPTARG
      ;;
   p) parallel=$OPTARG
      ;;
   c) csim="csim=1"
      ;;
   s) synth="synth=1"
      ;;
   r) cosim="cosim=1"
      ;;
   t) validation="validation=1"
      ;;
   l) vsynth="vsynth=1"
      ;;
   e) export="export=1"
      ;;
   n) reset="reset=1"
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
   dir="${filename}-${vivadover}"
   tarpath=`tar -tf "${archive}" | grep -m1 "${filename}"`
   slashes="${tarpath//[^\/]}"
   mkdir -p "${dir}" && tar -xzf "${archive}" -C "${dir}" --strip-components ${#slashes}
done

source ${vivadodir}/Vivado/${vivadover}/settings64.sh

opt="${reset} ${csim} ${synth} ${cosim} ${validation} ${vsynth} ${export}"

if [ "${parallel}" -gt 1 ]; then
   # Run in parallel
   (
   for dir in *-${vivadover}/ ; do
      ((n=n%parallel)); ((n++==0)) && wait
      run_vivado "${dir}" "${opt}" &
   done
   wait
   )
else
   # Run sequentially
   for dir in *-${vivadover}/ ; do
      run_vivado "${dir}" "${opt}"
   done
fi

# Check for build errors
for dir in *-${vivadover}/ ; do
   check_status "${dir}" "${opt}"
done

#cd "${rundir}"

exit ${failed}
