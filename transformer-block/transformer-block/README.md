# Transformer Block Inference Acceleration

This project implements a Transformer block for inference acceleration using both CPU and GPU. The implementation is designed to optimize the performance of various operations involved in the Transformer architecture.

## Project Structure

- **src/**: Contains the source code for both CPU and CUDA implementations.
  - **cpu/**: 
    - `transformer_cpu.cpp`: Implementation of the Transformer block on CPU.
    - `operations_cpu.cpp`: Implementation of various operations (matrix multiplication, softmax, layer normalization, etc.) on CPU.
    - `operations_cpu.h`: Header file containing function declarations for CPU operations.
  - **cuda/**: 
    - `transformer_cuda.cu`: Implementation of the Transformer block using CUDA for GPU execution.
    - `kernels.cu`: Implementation of CUDA kernels for GPU operations.
    - `kernels.cuh`: Header file containing declarations for CUDA kernels.
  - **common/**: 
    - `transformer_common.h`: Common definitions and headers used by both CPU and CUDA implementations.
    - `utils.cpp`: Implementation of utility functions for data loading and timing measurements.
    - `utils.h`: Header file containing declarations for utility functions.
  - `main.cpp`: Entry point of the program that manages execution on both CPU and GPU.

- **data/**: Contains binary files for input embeddings and weights.
  - `input_embeddings.bin`: Input embedding data.
  - `weights_q.bin`: Query weights.
  - `weights_k.bin`: Key weights.
  - `weights_v.bin`: Value weights.
  - `ff_weights1.bin`: Weights for the first feed-forward network layer.
  - `ff_weights2.bin`: Weights for the second feed-forward network layer.

- **out/**: Directory to store output layer values.

- **result/**: Directory to store execution time and error information.

- **Makefile**: Build configuration file for the project, including settings for SM_90.

- **run.sh**: Script to execute jobs using qsub.

## Usage

1. **Build the Project**: Navigate to the project root directory and run `make` to compile the source code.
2. **Prepare Input Data**: Ensure that the input data files are placed in the `data/` directory.
3. **Run the Program**: Execute the `run.sh` script to start the inference process on both CPU and GPU.
4. **Check Results**: Output values will be stored in the `out/` directory, and execution logs will be available in the `result/` directory.

## Performance Comparison

This project aims to compare the execution time of the Transformer block operations on CPU (optimized with OpenMP) and GPU (using CUDA). The results will help in understanding the performance benefits of using GPU for deep learning inference tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.