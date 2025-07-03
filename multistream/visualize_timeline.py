import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.patches as patches
import numpy as np
import sys
import re
import os

def parse_timeline_data(filename):
    """タイムラインデータをパースする"""
    timeline_data = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # TIMELINE_DATA_START と TIMELINE_DATA_END の間のデータを抽出
    match = re.search(r'=== TIMELINE_DATA_START ===(.*?)=== TIMELINE_DATA_END ===', 
                     content, re.DOTALL)
    
    if not match:
        print("タイムラインデータが見つかりませんでした")
        return []
    
    data_section = match.group(1)
    
    for line in data_section.strip().split('\n'):
        if line.startswith('BATCH,'):
            parts = line.split(',')
            batch_id = int(parts[1])
            stream_id = int(parts[2])
            start_time = float(parts[3])
            h2d_time = float(parts[4])
            kernel_time = float(parts[5])
            d2h_time = float(parts[6])
            end_time = float(parts[7])
            
            timeline_data.append({
                'batch': batch_id,
                'stream': stream_id,
                'start': start_time,
                'h2d_end': h2d_time,
                'kernel_end': kernel_time,
                'd2h_end': d2h_time,
                'end': end_time
            })
    
    return timeline_data

def create_timeline_diagram(timeline_data, output_file):
    """タイムラインダイヤグラムを作成"""
    if not timeline_data:
        print("データがありません")
        return
    
    # ストリーム数を取得
    num_streams = max(data['stream'] for data in timeline_data) + 1
    
    # 図のサイズを設定
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 色の設定
    colors = {
        'h2d': '#FF6B6B',      # 赤系（H2D転送）
        'kernel': '#4ECDC4',   # 青緑系（カーネル実行）
        'd2h': '#45B7D1',      # 青系（D2H転送）
        'idle': '#F0F0F0'      # グレー（アイドル）
    }
    
    # 各ストリームに対してタイムラインを描画
    for stream_id in range(num_streams):
        stream_data = [data for data in timeline_data if data['stream'] == stream_id]
        stream_data.sort(key=lambda x: x['start'])
        
        y_pos = stream_id * 0.8  # ストリーム間の間隔
        
        for data in stream_data:
            batch_id = data['batch']
            
            # H2D転送
            h2d_rect = patches.Rectangle(
                (data['start'], y_pos), 
                data['h2d_end'] - data['start'], 
                0.6,
                facecolor=colors['h2d'], 
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(h2d_rect)
            
            # カーネル実行
            kernel_rect = patches.Rectangle(
                (data['h2d_end'], y_pos), 
                data['kernel_end'] - data['h2d_end'], 
                0.6,
                facecolor=colors['kernel'], 
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(kernel_rect)
            
            # D2H転送
            d2h_rect = patches.Rectangle(
                (data['kernel_end'], y_pos), 
                data['d2h_end'] - data['kernel_end'], 
                0.6,
                facecolor=colors['d2h'], 
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(d2h_rect)
            
            # バッチ番号を表示
            ax.text(data['start'] + (data['end'] - data['start']) / 2, 
                   y_pos + 0.3, 
                   f'B{batch_id}', 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold')
    
    # 軸の設定
    ax.set_xlim(0, max(data['end'] for data in timeline_data) * 1.1)
    ax.set_ylim(-0.5, num_streams * 0.8)
    
    # Y軸のラベル
    ax.set_yticks([i * 0.8 + 0.3 for i in range(num_streams)])
    ax.set_yticklabels([f'Stream {i}' for i in range(num_streams)])
    
    # X軸の設定
    ax.set_xlabel('time (ms)', fontsize=12)
    ax.set_ylabel('stream', fontsize=12)
    ax.set_title('CUDA stream execution timeline', fontsize=14, fontweight='bold')
    
    # 凡例
    legend_elements = [
        patches.Patch(facecolor=colors['h2d'], label='H2D copy'),
        patches.Patch(facecolor=colors['kernel'], label='kernel launch'),
        patches.Patch(facecolor=colors['d2h'], label='D2H copy')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # グリッドの追加
    ax.grid(True, alpha=0.3)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"タイムラインダイヤグラムを {output_file} に保存しました")
    
    # 統計情報を表示
    print_statistics(timeline_data)

def print_statistics(timeline_data):
    """統計情報を表示"""
    print("\n=== 統計情報 ===")
    
    # 全体の実行時間
    total_time = max(data['end'] for data in timeline_data)
    print(f"全体実行時間: {total_time:.3f} ms")
    
    # 各フェーズの平均時間
    h2d_times = [data['h2d_end'] - data['start'] for data in timeline_data]
    kernel_times = [data['kernel_end'] - data['h2d_end'] for data in timeline_data]
    d2h_times = [data['d2h_end'] - data['kernel_end'] for data in timeline_data]
    
    print(f"H2D転送平均時間: {np.mean(h2d_times):.3f} ms")
    print(f"カーネル実行平均時間: {np.mean(kernel_times):.3f} ms")
    print(f"D2H転送平均時間: {np.mean(d2h_times):.3f} ms")
    
    # 並列度の計算
    num_streams = max(data['stream'] for data in timeline_data) + 1
    sequential_time = sum(data['end'] - data['start'] for data in timeline_data)
    parallel_efficiency = sequential_time / total_time
    
    print(f"並列効率: {parallel_efficiency:.2f}x")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python visualize_timeline.py <result_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    timeline_data = parse_timeline_data(input_file)
    
    if timeline_data:
        # 出力ファイル名を自動生成
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_timeline_diagram.png"
        create_timeline_diagram(timeline_data, output_file)
    else:
        print("タイムラインデータの解析に失敗しました")

if __name__ == "__main__":
    main()