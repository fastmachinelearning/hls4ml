#!/usr/bin/env bash
set -euo pipefail

APPIMAGE="bambu"

TMPINFO=""
MOUNT_PID=""
MOUNT_DIR=""

cleanup() {
    if [ -n "${MOUNT_PID:-}" ]; then kill "$MOUNT_PID" 2>/dev/null || true; fi
    if [ -n "${TMPINFO:-}" ]; then rm -f "$TMPINFO" || true; fi
}
trap cleanup EXIT

# 1. Use BAMBU_SQUASHFS_ROOT if set by setup-bambu.sh (extracted AppImage)
if [ -n "${BAMBU_SQUASHFS_ROOT:-}" ] && [ -d "$BAMBU_SQUASHFS_ROOT" ]; then
    MOUNT_DIR="$BAMBU_SQUASHFS_ROOT"
fi

# 2. Try --appimage-mount (actual AppImage on PATH)
if [ -z "$MOUNT_DIR" ] && command -v "$APPIMAGE" >/dev/null 2>&1; then
    TMPINFO="$(mktemp)"
    "$APPIMAGE" --appimage-mount >"$TMPINFO" 2>&1 &
    MOUNT_PID=$!

    for _ in {1..100}; do
        if [ -s "$TMPINFO" ]; then
            MOUNT_DIR=$(sed -n '1p' "$TMPINFO" | tr -d '\r\n')
            if [ -d "$MOUNT_DIR" ]; then break; fi
        fi
        sleep 0.05
    done

    if [ ! -d "$MOUNT_DIR" ]; then
        MOUNT_DIR=""
        MOUNT_PID=""
    fi
fi

# 3. Derive from extracted squashfs via which bambu → up 3 dirs
if [ -z "$MOUNT_DIR" ]; then
    BIN_PATH="$(which "$APPIMAGE" 2>/dev/null || true)"
    if [ -n "$BIN_PATH" ]; then
        APPDIR="$(dirname "$(dirname "$(dirname "$BIN_PATH")")")"
        if [ -x "$APPDIR/usr/bin/clang++-16" ] || \
           [ -x "$APPDIR/usr/compilers/clang-16/bin/clang++-16" ]; then
            MOUNT_DIR="$APPDIR"
        fi
    fi
fi

# Locate clang++-16 — required, no fallback to g++
CC=""
if [ -n "$MOUNT_DIR" ]; then
    for candidate in \
        "$MOUNT_DIR/usr/compilers/clang-16/bin/clang++-16" \
        "$MOUNT_DIR/usr/bin/clang++-16"
    do
        if [ -x "$candidate" ]; then CC="$candidate"; break; fi
    done
fi
if [ -z "$CC" ]; then
    echo "ERROR: Bambu clang++-16 not found. Set BAMBU_SQUASHFS_ROOT or ensure bambu AppImage is on PATH." >&2
    exit 1
fi

echo "Using compiler: $($CC --version | head -n1)"

CFLAGS="-O3 -fPIC"

# Include -std=c++23 if the compiler supports it (enables half and bfloat16 types, errors otherwise)
if echo "" | $CC -Werror -fsyntax-only -std=c++23 -xc++ - -o /dev/null &>/dev/null; then
    CFLAGS+=" -std=c++23"
else
    CFLAGS+=" -std=c++14"
fi

# Include -fno-gnu-unique if it is there
if echo "" | $CC -Werror -fsyntax-only -fno-gnu-unique -xc++ - -o /dev/null &>/dev/null; then
    CFLAGS+=" -fno-gnu-unique"
fi

LDFLAGS=""
INCFLAGS="-isystem ${MOUNT_DIR}/usr/include/panda"
PROJECT="myproject"
LIB_STAMP="mystamp"
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""

$CC $CFLAGS $INCFLAGS -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
$CC $CFLAGS $INCFLAGS -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}_float.cpp -o ${PROJECT}_float.o
$CC $CFLAGS $INCFLAGS -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${PROJECT}_float_test.cpp -o ${PROJECT}_float_test.o
$CC ${PROJECT}.o ${PROJECT}_float.o ${PROJECT}_float_test.o -o ${PROJECT}-${LIB_STAMP}_float_tb.exe

rm -f *.o

echo "Executable built: ${PROJECT}-${LIB_STAMP}_float_tb.exe"
