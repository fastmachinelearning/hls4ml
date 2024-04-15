#!/bin/bash

failed=0
basedir=hls_prj
all=0

function print_usage {
   echo "Usage: `basename $0` [OPTION]"
   echo ""
   echo "Cleans up the projects in provided directory."
   echo ""
   echo "Options are:"
   echo "   -d DIR"
   echo "      Base directory where projects are located."
   echo "   -a"
   echo "      Remove all projects, even the failed ones."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":d:ah" opt; do
   case "$opt" in
   d) basedir=$OPTARG
      ;;
   a) all=1
      ;;
   h)
      print_usage
      exit
      ;;
   esac
done

if [ ! -d "${basedir}" ]; then
   echo "Specified directory '${basedir}' does not exist."
   exit 1
fi

if [ "${all}" -eq 1 ]; then
   rm -rf "${basedir}"
   exit $?
fi

#rundir=`pwd`

cd "${basedir}"

rm -f *.tar.gz

# Delete
for dir in */ ; do
   if [ ! -f "${dir}BUILD_FAILED" ]; then
      rm -rf "${dir}"
      if [ $? -eq 0 ]; then
         echo "Removed ${dir%/}."
      else
         failed=1
      fi
   fi
done

#cd "${rundir}"

exit ${failed}
