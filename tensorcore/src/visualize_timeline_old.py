import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.patches as patches
import numpy as np
import sys
import re
import os

def parse_tensorcore_timeline_data(filename):
    """TensorCoreタイムラインデータをパースする"""
    timeline_data = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # TIMELINE_DATA_START と TIMELINE_DATA_END の間のデータを抽出
    match = re.search(r'=== TIMELINE_DATA_START ===(.*?)=== TIMELINE_DATA_END ===', 
                     content, re.DOTALL)
    
    if not match:
        print("TensorCoreタイムラインデータが見つかりませんでした")
        return []
    
    data_section = match.group(1)
    
    for line in data_section.strip().split('\n'):
        if line.startswith('PHASE,'):
            parts = line.split(',')
            phase_type = parts[0]  # 'PHASE'
            method = parts[1]      # 'Global' or 'Shared'
            phase_name = parts[2]  # フェーズ名
            start_time = float(parts[3]) * 1000  # msをμsに変換
            end_time = float(parts[4]) * 1000    # msをμsに変換
            
            timeline_data.append({
                'method': method,
                'phase': phase_name,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
    
    return timeline_data

def create_tensorcore_timeline_diagram(timeline_data, output_file):
    """TensorCoreタイムラインダイヤグラムを作成"""
    if not timeline_data:
        print("データがありません")
        return
    
    # 図のサイズを設定（2つのサブプロットに分割）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # 色の設定（フェーズ別）
    phase_colors = {
        'Fragment_Declare': '#FF6B6B',           # 赤系（フラグメント宣言）
        'Global_to_Frag': '#4ECDC4',     # 青緑系（グローバル→フラグメント）
        'Global_to_Shared': '#45B7D1',   # 青系（グローバル→共有メモリ）
        'Shared_to_Frag': '#96CEB4',     # 緑系（共有メモリ→フラグメント）
        'Compute': '#FFEAA7',            # 黄系（計算）
        'Store': '#DDA0DD',              # 紫系（ストア）
        'Fragment_Init': '#FD79A8'            # ピンク系（フラグメント初期化）
    }
    
    # メソッド別にデータを分類
    global_data = [d for d in timeline_data if d['method'] == 'Global']
    shared_data = [d for d in timeline_data if d['method'] == 'Shared']
    
    # 全体の最大時間を計算（横軸のスケール統一のため）
    all_end_times = [d['end'] for d in timeline_data]
    max_time = max(all_end_times) if all_end_times else 1
    
    # Global Memory版のタイムライン（上のサブプロット）
    draw_timeline(ax1, global_data, phase_colors, 'Global Memory版 TensorCore実行タイムライン', max_time)
    
    # Shared Memory版のタイムライン（下のサブプロット）
    draw_timeline(ax2, shared_data, phase_colors, 'Shared Memory版 TensorCore実行タイムライン', max_time)
    
    # 全体の凡例を追加
    legend_elements = [
        patches.Patch(facecolor=color, label=phase.replace('_', '→'))
        for phase, color in phase_colors.items()
    ]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # 性能比較の注釈を追加
    global_total = sum(d['duration'] for d in global_data)
    shared_total = sum(d['duration'] for d in shared_data)
    speedup = global_total / shared_total if shared_total > 0 else 1.0
    
    fig.suptitle(f'TensorCore実行タイムライン比較\n'
                f'Global: {global_total:.1f}μs, Shared: {shared_total:.1f}μs, '
                f'速度向上: {speedup:.2f}x', 
                fontsize=16, fontweight='bold')
    
    # 保存
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # 凡例のスペースを確保
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"TensorCoreタイムラインダイヤグラムを {output_file} に保存しました")
    
    # 統計情報を表示
    print_tensorcore_statistics(timeline_data)

def draw_timeline(ax, data_list, phase_colors, title, max_time):
    """単一のタイムラインを描画"""
    if not data_list:
        ax.set_title(f"{title} - データなし")
        return
    
    y_pos = 0.3  # 単一の行に描画
    
    for data in data_list:
        phase = data['phase']
        color = phase_colors.get(phase, '#CCCCCC')
        
        # フェーズの矩形を描画
        rect = patches.Rectangle(
            (data['start'], y_pos), 
            data['duration'], 
            0.4,
            facecolor=color, 
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # フェーズ名とタイミングを表示
        if data['duration'] > 1000:  # 1000μs以上の場合のみ矩形内に表示
            ax.text(data['start'] + data['duration'] / 2, 
                   y_pos + 0.2, 
                   f'{phase}\n{data["duration"]:.3f}μs', 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        else:
            # 短いフェーズの場合は上側に表示
            ax.text(data['start'] + data['duration'] / 2, 
                   y_pos + 0.6, 
                   f'{phase}: {data["duration"]:.3f}μs', 
                   ha='center', va='bottom', 
                   fontsize=8,
                   rotation=45)
    
    # 軸の設定（横軸のスケールを統一）
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(0, 1)
    
    # Y軸を非表示
    ax.set_yticks([])
    
    # X軸の設定
    ax.set_xlabel('時間 (μs)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # グリッドの追加
    ax.grid(True, alpha=0.3, axis='x')
    
    # 各フェーズの統計を表示
    total_time = sum(d['duration'] for d in data_list)
    stats_text = f'総時間: {total_time:.3f}μs\n'
    for data in data_list:
        percentage = (data['duration'] / total_time) * 100 if total_time > 0 else 0
        stats_text += f'{data["phase"]}: {percentage:.1f}%\n'
    
    ax.text(0.02, 0.98, stats_text.strip(),
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)

def print_tensorcore_statistics(timeline_data):
    """TensorCore統計情報を表示"""
    print("\n=== TensorCore 詳細統計 ===")
    
    # メソッド別統計
    methods = ['Global', 'Shared']
    method_totals = {}
    
    for method in methods:
        method_data = [d for d in timeline_data if d['method'] == method]
        total_time = sum(d['duration'] for d in method_data)
        method_totals[method] = total_time
        
        print(f"\n{method} Memory方式:")
        print(f"  総時間: {total_time:.3f} μs")
        
        for data in method_data:
            percentage = (data['duration'] / total_time) * 100 if total_time > 0 else 0
            print(f"  {data['phase']}: {data['duration']:.3f} μs ({percentage:.1f}%)")
    
    # 全体の比較
    if method_totals['Global'] > 0 and method_totals['Shared'] > 0:
        speedup = method_totals['Global'] / method_totals['Shared']
        print(f"\n=== 全体比較 ===")
        print(f"速度向上: {speedup:.3f}x (Shared Memory版が高速)")
    
    # フェーズ別比較
    print("\n=== フェーズ別比較 (Global/Shared比) ===")
    global_phases = {d['phase']: d['duration'] for d in timeline_data if d['method'] == 'Global'}
    shared_phases = {d['phase']: d['duration'] for d in timeline_data if d['method'] == 'Shared'}
    
    for phase in set(global_phases.keys()) & set(shared_phases.keys()):
        if shared_phases[phase] > 0:
            ratio = global_phases[phase] / shared_phases[phase]
            print(f"  {phase}: {ratio:.2f}x")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python visualize_timeline.py <result_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    timeline_data = parse_tensorcore_timeline_data(input_file)
    
    if timeline_data:
        # 出力ファイル名を自動生成
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_tensorcore_timeline_diagram.png"
        create_tensorcore_timeline_diagram(timeline_data, output_file)
    else:
        print("TensorCoreタイムラインデータの解析に失敗しました")

if __name__ == "__main__":
    main()