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
ORIGINAL_PROJECT=myproject
PROJECT=myproject_stitched
LIB_STAMP=mystamp
BASEDIR="$(cd "$(dirname "$0")" && cd .. && pwd)"
AP_TYPES_PATH="-I${BASEDIR}/${graph_project_names[0]}/firmware/ap_types/"
INCFLAGS=""
OUTPUT_DIR="${BASEDIR}/stitched/firmware"

mkdir -p "${OUTPUT_DIR}"

# Compile all graphs
OBJECT_FILES=()
for g in "${graph_project_names[@]}"; do
    WEIGHTS_DIR="\"${BASEDIR}/${g}/firmware/weights\""
    SRC_FILE="${g}/firmware/${ORIGINAL_PROJECT}_${g}.cpp"
    OBJ_FILE="${ORIGINAL_PROJECT}_${g}.o"
    
    ${CC} ${CFLAGS} ${AP_TYPES_PATH} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c "${BASEDIR}/${SRC_FILE}" -o "${OBJ_FILE}"
    OBJECT_FILES+=("${OBJ_FILE}")
    INCFLAGS+="-I${BASEDIR}/${g}/ "
done

${CC} ${CFLAGS} ${INCFLAGS} ${AP_TYPES_PATH} -c "${PROJECT}_bridge.cpp" -o ${PROJECT}_bridge.o

${CC} ${CFLAGS} ${INCFLAGS} ${AP_TYPES_PATH} -shared "${OBJECT_FILES[@]}" ${PROJECT}_bridge.o -o "${OUTPUT_DIR}/${PROJECT}-${LIB_STAMP}.so"

rm -f "${OBJECT_FILES[@]}"
rm -f ${PROJECT}_bridge.o
