#!/bin/bash

failed=0
basedir=vivado_prj
remove=0

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Checks if project files generated with Python 2 and 3 differ."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory where projects are located."
   echo "   -r 2|3"
   echo "      Remove Python 2 or 3 projects and keep others. If not specified,"
   echo "      nothing is removed. Projects that differ are not removed."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":d:r:h" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   r) remove=$OPTARG
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

py2dirs=()
for dir in *-python2-*/ ; do
   py2dirs+=("${dir}")
done

py3dirs=()
for dir in *-python3-*/ ; do
   py3dirs+=("${dir}")
done

if [ "${#py2dirs[@]}" -eq "${#py3dirs[@]}" ]; then
   echo "Found ${#py2dirs[@]} project(s)."
else
   echo "Found ${#py2dirs[@]} Python 2 project(s) and ${#py3dirs[@]} Python 3 project(s). Exiting."
   exit 1
fi

for i in ${!py2dirs[*]}; do
   py2dir="${py2dirs[$i]}"
   py3dir="${py3dirs[$i]}"
   echo "Checking ${py2dir%/} and ${py3dir%/}:"
   diff -rq "${py2dir}" "${py3dir}"
   if [ $? -eq 0 ]; then
      echo "No differences found."
      rm -rf "${py2dir}" "${py3dir}" 
      if [ "${remove}" -eq 2 ]; then
         rm -f "${py2dir%/}.tar.gz"
         echo "Removed ${py3dir%/}"
      fi
      if [ "${remove}" -eq 3 ]; then
         rm -f "${py3dir%/}.tar.gz"
         echo "Removed ${py2dir%/}"
      fi
   else
      diff -r "${py2dir}" "${py3dir}"
      failed=1
   fi
   echo ""
done

#cd "${rundir}"

exit ${failed}
