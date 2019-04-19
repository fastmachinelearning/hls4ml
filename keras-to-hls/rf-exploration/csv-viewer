#!/bin/bash

if [ -z "$1" ]; then
    echo "USAGE: csv-viewer <file.csv>"
    return 1
fi

CSV=$1

column -s, -t < $CSV | less -#2 -N -S
