
# 使用方法チェック
if [ $# -ne 1 ]; then
    echo "使用方法: $0 <timestamp>"
    echo "例: $0 20250703111621"
    echo "これは result_20250703111621.txt を処理します"
    exit 1
fi

# 引数からファイル名を構築
TIMESTAMP=$1
RESULT_FILE="result/result_matmul_multistream_${TIMESTAMP}.txt"

# ファイルの存在確認
if [ ! -f "$RESULT_FILE" ]; then
    echo "エラー: ファイル '$RESULT_FILE' が見つかりません"
    echo "現在のディレクトリのresult_*.txtファイル:"
    ls -1 result_*.txt 2>/dev/null || echo "  result_*.txt ファイルが見つかりません"
    exit 1
fi

echo "処理するファイル: $RESULT_FILE"

# 仮想環境の確認・有効化
if [ ! -d "venv" ]; then
    echo "仮想環境が見つかりません。作成します..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Python可視化スクリプトの実行
python visualize_timeline.py "$RESULT_FILE"

echo "可視化が完了しました"

deactivate

