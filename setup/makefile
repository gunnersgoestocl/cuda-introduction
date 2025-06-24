CC := gcc
NVCC := nvcc
CFLAGS := -Wall -Wextra
NVCCFLAGS := -arch=sm_90 -O3
# NVCCFLAGS := -arch=sm_90 -O3	# 逐次プログラム
# NVCCFLAGS := -arch=sm_90 -O3 -Xcompiler -fopenmp	# OpenMP並列化
# NVCCFLAGS := -arch=sm_90 -O3 -I$OMPI_INC -L$LIBRARY_PATH -lmpi	# MPI並列化


# .c ファイルから実行ファイルを生成
C_EXECUTABLES := $(patsubst %.c,%,$(wildcard *.c))

# .cu ファイルから実行ファイルを生成
CUDA_EXECUTABLES := $(patsubst %.cu,%,$(wildcard *.cu))

# デフォルトのターゲット
all: $(C_EXECUTABLES) $(CUDA_EXECUTABLES)

# .c ファイルから実行ファイルを生成するルール
%: %.c
	$(CC) $(CFLAGS) $< -o $@

# .cu ファイルから実行ファイルを生成するルール
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# clean ターゲットの定義
.PHONY: clean
clean:
	rm -f $(C_EXECUTABLES) $(CUDA_EXECUTABLES)