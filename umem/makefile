NVCC        ?= nvcc
ARCH        ?= sm_90
SRC         := vecadd_compare.cu
NVCCFLAGS   := -O3 -arch=$(ARCH) -lineinfo -use_fast_math

# デフォルトターゲット
all: um explicit

# Unified Memory 版
um: $(SRC)
	$(NVCC) $(NVCCFLAGS) -DUSE_MANAGED $< -o $@

# Explicit Copy 版
explicit: $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f um explicit *.o
.PHONY: all clean
