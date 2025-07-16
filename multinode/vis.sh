#!/bin/bash

# 使用方法チェック
if [ $# -ne 1 ]; then
    echo "使用方法: $0 <timestamp>"
    echo "例: $0 20250708172351"
    echo "これは timing_data_20250708172351.csv を処理します"
    exit 1
fi

# 引数からファイル名を構築
TIMESTAMP=$1
# RESULT_FILE="result/result_${TIMESTAMP}.txt"
CSV_FILE="result/timing_data_${TIMESTAMP}.csv"

# ファイルの存在確認
if [ ! -f "$CSV_FILE" ]; then
    echo "エラー: ファイル '$CSV_FILE' が見つかりません"
    echo "現在のディレクトリのresult_*.txtファイル:"
    ls -1 result/result_*.txt 2>/dev/null || echo "  result_*.txt ファイルが見つかりません"
    exit 1
fi

echo "処理するファイル: $CSV_FILE"

# 仮想環境の確認・有効化
if [ ! -d "venv" ]; then
    echo "仮想環境を作成中..."
    python -m venv venv
    source venv/bin/activate
	# pip install pandas matplotlib
	# pip freeze > requirements.txt
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Python可視化スクリプトの実行
python visualize_timing.py "$CSV_FILE"

echo "Multi-Nodeの可視化が完了しました"