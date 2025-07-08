NVCC := nvcc
CFLAGS := -O3 -arch=sm_90 -std=c++17

# ソースファイル
SRC := tensor_core_nv.cu

# 出力ファイル名
TARGET := tensor_core_nv

# タイムスタンプの生成
TS := $(shell date +"%Y%m%d%H%M")

# 実行ファイルのビルド
all: $(TARGET) extract-ptx

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@

# PTXとSASSの抽出（ビルド後に自動実行）
extract-ptx: $(TARGET)
	cuobjdump -ptx $(TARGET) > $(TARGET)_$(TS).ptx
	cuobjdump --dump-sass $(TARGET) > $(TARGET)_$(TS).sass
	@echo "PTXファイル生成完了: $(TARGET)_$(TS).ptx"
	@echo "SASSファイル生成完了: $(TARGET)_$(TS).sass"
	@echo "TensorCore関連命令の検索結果:"
	@grep -i "mma\|wmma" $(TARGET)_$(TS).ptx || echo "PTXにTensorCore命令が見つかりません"

# 実行用ターゲット
run: $(TARGET)
	./$(TARGET)

# クリーンアップ
clean:
	rm -f $(TARGET) *.o *.ptx *.sass

.PHONY: all extract-ptx run clean