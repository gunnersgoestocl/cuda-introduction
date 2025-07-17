# Makefile for Transformer Block Project

# Compiler and flags
# CXX = clang++
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -fopenmp -I./src/common
NVCCFLAGS = -O3 -arch=sm_90

# Directories
SRC_DIR = src
CPU_DIR = $(SRC_DIR)/cpu
CUDA_DIR = $(SRC_DIR)/cuda
COMMON_DIR = $(SRC_DIR)/common
OUT_DIR = out
RESULT_DIR = result

# Source files
CPU_SRC = $(CPU_DIR)/transformer_cpu.cpp $(CPU_DIR)/operations_cpu.cpp
CUDA_SRC = $(CUDA_DIR)/transformer_cuda.cu $(CUDA_DIR)/kernels.cu
COMMON_SRC = $(COMMON_DIR)/utils.cpp

# Executables
# CPU_EXEC = $(OUT_DIR)/transformer_cpu
# CUDA_EXEC = $(OUT_DIR)/transformer_cuda
CPU_EXEC = cpu_transformer
CUDA_EXEC = cuda_transformer

# Targets
all: $(CPU_EXEC) $(CUDA_EXEC)

$(CPU_EXEC): $(CPU_SRC) $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(CUDA_EXEC): $(CUDA_SRC) $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(OUT_DIR)/* $(RESULT_DIR)/* $(CPU_EXEC) $(CUDA_EXEC)

.PHONY: all clean