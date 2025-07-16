#!/bin/bash

# Load necessary modules (if applicable)
# module load gcc/9.3.0
# module load cuda/11.2

# Set variables
CPU_EXECUTABLE="cpu_transformer"
CUDA_EXECUTABLE="cuda_transformer"
RESULT_DIR="./result"
OUT_DIR="./out"

# Create output and result directories if they do not exist
mkdir -p $RESULT_DIR
mkdir -p $OUT_DIR

# Run CPU version
echo "Running CPU version..."
./$CPU_EXECUTABLE > $RESULT_DIR/cpu_output.log 2> $RESULT_DIR/cpu_error.log

# Check for errors in CPU execution
if [ $? -ne 0 ]; then
    echo "CPU execution failed. Check the log for details."
    exit 1
fi

# Run CUDA version
echo "Running CUDA version..."
./$CUDA_EXECUTABLE > $RESULT_DIR/cuda_output.log 2> $RESULT_DIR/cuda_error.log

# Check for errors in CUDA execution
if [ $? -ne 0 ]; then
    echo "CUDA execution failed. Check the log for details."
    exit 1
fi

# Compare outputs
echo "Comparing outputs..."
diff $OUT_DIR/cpu_output.bin $OUT_DIR/cuda_output.bin > $RESULT_DIR/output_diff.log

if [ $? -eq 0 ]; then
    echo "Outputs match."
else
    echo "Outputs do not match. Check the diff log for details."
fi

echo "Execution completed."