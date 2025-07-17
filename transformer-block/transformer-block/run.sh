#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

# ジョブディレクトリに移動
cd $PBS_O_WORKDIR

# モジュールのロード
module purge
module load cuda

# Set variables (Makefileと一致)
CPU_EXECUTABLE="./cpu_transformer"
CUDA_EXECUTABLE="./cuda_transformer"
RESULT_DIR="./result"
OUT_DIR="./out"

echo "=== Transformer Block Performance Comparison ==="
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

# Create output and result directories if they do not exist
mkdir -p $RESULT_DIR
mkdir -p $OUT_DIR

echo "Building executables..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

echo "Build completed successfully."

# Run CPU version
echo ""
echo "=== Running CPU version ==="
if [ -x "$CPU_EXECUTABLE" ]; then
    $CPU_EXECUTABLE > $RESULT_DIR/cpu_output.log 2> $RESULT_DIR/cpu_error.log
    CPU_EXIT_CODE=$?
    
    if [ $CPU_EXIT_CODE -ne 0 ]; then
        echo "CPU execution failed with exit code $CPU_EXIT_CODE"
        echo "Error log:"
        cat $RESULT_DIR/cpu_error.log
        exit 1
    else
        echo "CPU execution completed successfully"
        echo "CPU output log saved to: $RESULT_DIR/cpu_output.log"
    fi
else
    echo "CPU executable not found: $CPU_EXECUTABLE"
    exit 1
fi

# Run CUDA version (if it exists)
echo ""
echo "=== Running CUDA version ==="
if [ -x "$CUDA_EXECUTABLE" ]; then
    $CUDA_EXECUTABLE > $RESULT_DIR/cuda_output.log 2> $RESULT_DIR/cuda_error.log
    CUDA_EXIT_CODE=$?
    
    if [ $CUDA_EXIT_CODE -ne 0 ]; then
        echo "CUDA execution failed with exit code $CUDA_EXIT_CODE"
        echo "Error log:"
        cat $RESULT_DIR/cuda_error.log
        # CUDAが失敗してもCPU結果は有効なので続行
    else
        echo "CUDA execution completed successfully"
        echo "CUDA output log saved to: $RESULT_DIR/cuda_output.log"
        
        # Compare outputs (両方が成功した場合のみ)
        echo ""
        echo "=== Comparing outputs ==="
        if [ -f "$RESULT_DIR/cpu_output.bin" ] && [ -f "$RESULT_DIR/cuda_output.bin" ]; then
            diff $RESULT_DIR/cpu_output.bin $RESULT_DIR/cuda_output.bin > $RESULT_DIR/output_diff.log 2>&1
            
            if [ $? -eq 0 ]; then
                echo "✓ Outputs match perfectly"
            else
                echo "⚠ Outputs differ (expected for different implementations)"
                echo "Difference log saved to: $RESULT_DIR/output_diff.log"
            fi
        else
            echo "Output files not found for comparison"
        fi
    fi
else
    echo "CUDA executable not found: $CUDA_EXECUTABLE (this is OK if CUDA version is not implemented yet)"
fi

# Display results summary
echo ""
echo "=== Execution Summary ==="
echo "Job completed at: $(date)"

if [ -f "$RESULT_DIR/cpu_output.log" ]; then
    echo "CPU execution log:"
    tail -10 $RESULT_DIR/cpu_output.log
fi

if [ -f "$RESULT_DIR/cuda_output.log" ]; then
    echo ""
    echo "CUDA execution log:"
    tail -10 $RESULT_DIR/cuda_output.log
fi

echo ""
echo "All logs saved in: $RESULT_DIR/"
echo "Execution completed successfully."