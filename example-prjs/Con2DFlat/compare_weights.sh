#!/bin/bash

folder1=$(pwd)/my-Catapult-test/firmware/weights
folder2=$(pwd)/my-Vivado-test/firmware/weights

# Loop through all .txt files in the first folder
for file in "$folder1"/*.txt; do
    filename=$(basename "$file")
    file2="$folder2/$filename"

    # Check if corresponding file exists in the second folder
    if [ -f "$file2" ]; then
        echo "Comparing $file and $file2"
        diff_output=$(diff "$file" "$file2")

        # Check if files are identical or have differences
        if [ -z "$diff_output" ]; then
            echo "Files are identical."
        else
            echo "Files have differences:"
            echo "$diff_output"
        fi
    else
        echo "$file2 not found in the second folder."
    fi
done