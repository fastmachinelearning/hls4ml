#!/bin/bash

failed=0
basedir=hls_prj
full=0
brief=0

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Prints synthesis reports found in projects in the provided directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory where projects are located."
   echo "   -b"
   echo "      Print only summary of performance and utilization estimates."
   echo "   -f"
   echo "      Print whole report."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":d:bfh" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   b) brief=1
      ;;
   f) full=1
      ;;
   h)
      print_usage
      exit
      ;;
   esac
done

if [ "${brief}" -eq "${full}" ]; then
   echo "Argument -b or -f must be provided."
   exit 1
fi

if [ ! -d "${basedir}" ]; then
   echo "Specified directory '${basedir}' does not exist."
   exit 1
fi

#rundir=`pwd`

cd "${basedir}"

for dir in */ ; do
   cd ${dir}
   prjdir="myproject_prj"
   prjname="myproject"
   for subdir in *_prj/ ; do
      prjdir=${subdir}
      prjname="${prjdir%_prj/}"
   done
   prjdir="${prjdir}solution1/syn/report"
   if [ -d "$prjdir" ]; then
      echo "Synthesis report for ${dir%/}"
      if [ "${brief}" -eq 1 ]; then
         sed "/* DSP48/Q" "${prjdir}/${prjname}_csynth.rpt"
      else
         cat "${prjdir}/${prjname}_csynth.rpt"
      fi
   else
      echo "No report files found in ${dir}."
      failed=1
   fi
   cd ..
done

#cd "${rundir}"

exit ${failed}
