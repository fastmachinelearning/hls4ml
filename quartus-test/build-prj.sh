#!/bin/bash

basedir=quartus_prj
parallel=1


function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Builds Quartus HLS projects found in the current directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory of projects to build. Defaults to 'quartus_prj'."
   echo "   -n"
   echo "      Create new project (reset any existing)."
   echo "   -h"
   echo "      Prints this help message."
}

function run_quartus {
   dir=$1
   echo "Building project in ${dir}"
   cd ${dir}
   cmd="make myproject-fpga"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch BUILD_FAILED
   fi
   cd ..
   return ${failed}
}

function run_simulation {
   dir=$1
   echo "Running sim in ${dir}"
   cd ${dir}
   cmd="./myproject-fpga"
   eval ${cmd}
   if [ $? -eq 1 ]; then
      touch SIM_FAILED
   fi
   cd ..
   return ${failed}
}

while getopts ":d:nh" opt; do
   case "$opt" in
   d) basedir=$OPTARG
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
   dir="${filename}-build"
   tarpath=`tar -tf "${archive}" | grep -m1 "${filename}"`
   slashes="${tarpath//[^\/]}"
   mkdir -p "${dir}" && tar -xzf "${archive}" -C "${dir}" --strip-components ${#slashes}
done

# Run sequentially
for dir in *-"build" ; do
   run_quartus "${dir}"
done

# Check for build errors
for dir in *-"build" ; do
   run_simulation "${dir}"
done

#cd "${rundir}"

exit ${failed}
