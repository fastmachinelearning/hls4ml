#!/bin/bash

failed=0
latency=0
utilization=0

function print_usage {
   echo "Usage: `basename $0` [OPTION] ORIGINAL_REPORT NEW_REPORT"
   echo ""
   echo "Compares two synthesis reports."
   echo ""
   echo "Options are:"
   echo "   -l"
   echo "      Compare latency."
   echo "   -u"
   echo "      Compare utilization estimates."
   echo "   -h"
   echo "      Prints this help message."
}

while getopts ":luh" opt; do
   case "$opt" in
   l) latency=1
      ;;
   u) utilization=1
      ;;
   h)
      print_usage
      exit
      ;;
   esac
done

shift $((OPTIND-1))

report_files=("$@")
if [[ ! ${#report_files[@]} -eq 2 ]]; then
   echo "Report files not specified."
   exit 1
fi

if [[ "${latency}" -eq 0 ]] && [[ "${util}" -eq 0 ]]; then
   echo "Argument -l or -u must be provided."
   exit 1
fi

original="${report_files[0]}"
new="${report_files[1]}"

rptname_orig=()
reports_orig=()
report=""
while IFS='' read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == "Synthesis report"* ]] && [[ "${report}" != "" ]]; then
        rptname_orig+=("${line}")
        reports_orig+=("${report}")
        report=""
    fi
    report+="${line}"$'\n'
done < "${original}"

rptname_new=()
reports_new=()
report=""
while IFS='' read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == "Synthesis report"* ]] && [[ "${report}" != "" ]]; then
        rptname_new+=("${line}")
        reports_new+=("${report}")
        report=""
    fi
    report+="${line}"$'\n'
done < "${new}"

for idx_orig in "${!rptname_orig[@]}"; do
   rptname="${rptname_orig[$idx_orig]}"
   idx_new="${idx_orig}"
   for j in "${!rptname_new[@]}"; do
      if [[ "${rptname_new[$j]}" = "${rptname}" ]]; then
         idx_new="${j}"
      fi
   done

   report_orig="${reports_orig[$idx_orig]}"
   report_new="${reports_new[$idx_new]}"

   if [ "${latency}" -eq 1 ]; then
      latency_orig=$(grep -A7 "+ Latency" <<< "${report_orig}")
      latency_new=$(grep -A7 "+ Latency" <<< "${report_new}")
      if [[ "${latency_orig}" != "${latency_new}" ]]; then
         failed=1
         echo "${rptname} has changed"
         echo ""
         left="Original:"$'\n'
         left+="${latency_orig}"
         right="New:"$'\n'
         right+="${latency_new}"
         column <(echo "${left}") <(echo "${right}")
         echo ""
         echo ""
         echo ""
      fi
   fi

   if [ "${utilization}" -eq 1 ]; then
      utilization_orig=$(grep -B3 -A13 "|DSP" <<< "${report_orig}")
      utilization_new=$(grep -B3 -A13 "|DSP" <<< "${report_new}")
      if [[ "${utilization_orig}" != "${utilization_new}" ]]; then
         failed=1
         echo "${rptname} has changed"
         echo ""
         left="Original:"$'\n'
         left+="${utilization_orig}"
         right="New:"$'\n'
         right+="${utilization_new}"
         column <(echo "${left}") <(echo "${right}")
         echo ""
         echo ""
         echo ""
      fi
   fi
done

exit ${failed}
