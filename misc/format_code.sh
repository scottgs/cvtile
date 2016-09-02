#!/bin/bash

####
# cvTile Code Formatter
####
# This script finds all relevant C++ and CUDA source files, and formats
# the code to adhere to the cvTile style rules configured within
# the file '.clang-format'
####
# Requires 'clang-format' tool be installed.
####

if hash clang-format 2>/dev/null; then
    # Lists all project folders containing C/C++/CUDA source files.
    for DIRECTORY in ../apps ../src ../tests ../src/algorithms ../src/base ../src/gpu ../src/gpu/drivers ../src/gpu/kernels
    do
        echo "Formatting code in folder $DIRECTORY/"
        find "$DIRECTORY" \( -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.cu' -or -name '*.cuh' \) -print0 | xargs -0 clang-format -i
    done
else
    # clang-format tool not installed
    echo "This script requires 'clang-format' be installed."
    echo "Exiting."
fi

