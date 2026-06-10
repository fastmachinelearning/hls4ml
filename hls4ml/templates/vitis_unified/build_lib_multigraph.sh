#!/bin/bash
set -e

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi

graph_project_names=(mygraph_name_list)

LDFLAGS=
VITIS_UNIFIED_FLAGS="VITIS_UNIFIED"
CFLAGS="$CFLAGS -D$VITIS_UNIFIED_FLAGS"

ORIGINAL_PROJECT=myproject
PROJECT=myproject_stitched
LIB_STAMP=mystamp
BASEDIR="$(cd "$(dirname "$0")" && cd .. && pwd)"
INCFLAGS=""
OUTPUT_DIR="${BASEDIR}/stitched/firmware"
WEIGHTS_DIR="\"${BASEDIR}/stitched/firmware/weights\""

mkdir -p "${OUTPUT_DIR}"

# Compile all graphs in parallel
OBJECT_FILES=()
PIDS=()

for g in "${graph_project_names[@]}"; do
    SRC_FILE="${g}/firmware/${ORIGINAL_PROJECT}_${g}.cpp"
    OBJ_FILE="${ORIGINAL_PROJECT}_${g}.o"
    AP_TYPES_PATH="-I${BASEDIR}/${g}/firmware/ap_types/"
    (
        ${CC} ${CFLAGS} ${AP_TYPES_PATH} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c "${BASEDIR}/${SRC_FILE}" -o "${OBJ_FILE}"
    ) &
    PIDS+=($!)
    OBJECT_FILES+=("${OBJ_FILE}")
    INCFLAGS+="-I${BASEDIR}/${g}/ "
done

# compile axi_stream as well

for g in "${graph_project_names[@]}"; do
    SRC_FILE="${g}/firmware/${ORIGINAL_PROJECT}_${g}_axi.cpp"
    OBJ_FILE="${ORIGINAL_PROJECT}_${g}_axi.o"
    AP_TYPES_PATH="-I${BASEDIR}/${g}/firmware/ap_types/"
    (
        ${CC} ${CFLAGS} ${AP_TYPES_PATH} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c "${BASEDIR}/${SRC_FILE}" -o "${OBJ_FILE}"
    ) &
    PIDS+=($!)
    OBJECT_FILES+=("${OBJ_FILE}")
    #INCFLAGS+="-I${BASEDIR}/${g}/ "
done



for pid in "${PIDS[@]}"; do
    wait $pid
done

AP_TYPES_PATH="-I${BASEDIR}/${graph_project_names[@]: -1}/firmware/ap_types/"

${CC} ${CFLAGS} ${INCFLAGS} ${AP_TYPES_PATH} -c "${PROJECT}_bridge.cpp" -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} ${AP_TYPES_PATH} -shared "${OBJECT_FILES[@]}" ${PROJECT}_bridge.o -o "${OUTPUT_DIR}/${PROJECT}-${LIB_STAMP}.so"

rm -f "${OBJECT_FILES[@]}"
rm -f ${PROJECT}_bridge.o
