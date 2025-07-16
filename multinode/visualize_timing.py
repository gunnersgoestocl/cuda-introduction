import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

def create_timing_visualization(csv_file):
    """
    分散計算のタイミングデータを可視化する
    """
    
    # データ読み込み
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_file}' が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return
    
    # 必要な列があるかチェック
    required_columns = ['rank', 'data_distribution', 'host_to_device', 
                       'kernel_execution', 'device_to_host', 'data_collection', 'total_time']
    
    # タイムスタンプ列も確認
    timestamp_columns = ['global_start', 'data_dist_start', 'data_dist_end',
                        'host_to_dev_start', 'host_to_dev_end', 'kernel_start', 'kernel_end',
                        'dev_to_host_start', 'dev_to_host_end', 'data_coll_start', 
                        'data_coll_end', 'global_end']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    missing_timestamps = [col for col in timestamp_columns if col not in df.columns]
    
    if missing_columns:
        print(f"エラー: 必要な列が不足しています: {missing_columns}")
        return
    
    # タイムスタンプ列がない場合はレガシーモードで実行
    has_timestamps = len(missing_timestamps) == 0
    
    # 基本統計情報を表示
    print("=== タイミングデータ統計 ===")
    print(f"ランク数: {len(df)}")
    print(f"最大総実行時間: {df['total_time'].max():.6f} 秒")
    print(f"最小総実行時間: {df['total_time'].min():.6f} 秒")
    print(f"平均総実行時間: {df['total_time'].mean():.6f} 秒")
    print(f"標準偏差: {df['total_time'].std():.6f} 秒")
    
    # 図のサイズとスタイル設定
    plt.style.use('default')
    
    if has_timestamps:
        # タイムスタンプデータがある場合は6つのサブプロット
        fig = plt.figure(figsize=(20, 12))
        subplot_layout = (2, 3)
    else:
        # タイムスタンプデータがない場合は5つのサブプロット
        fig = plt.figure(figsize=(16, 12))
        subplot_layout = (2, 3)
    
    # 1. 積み上げ棒グラフ（各ランクの時間内訳）
    ax1 = plt.subplot(subplot_layout[0], subplot_layout[1], 1)
    
    phases = ['data_distribution', 'host_to_device', 'kernel_execution', 
              'device_to_host', 'data_collection']
    phase_labels = ['data distribution', 'Host→Device', 'kernel execution', 
                   'Device→Host', 'result gathering']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bottom = np.zeros(len(df))
    for i, (phase, label, color) in enumerate(zip(phases, phase_labels, colors)):
        ax1.bar(df['rank'], df[phase], bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += df[phase]
    
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Time (sec)')
    ax1.set_title('Breakdown of processing time for each rank')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 総実行時間の比較
    ax2 = plt.subplot(subplot_layout[0], subplot_layout[1], 2)
    bars = ax2.bar(df['rank'], df['total_time'], color='#74B9FF', alpha=0.8)
    
    # 最大値と最小値にマーキング
    max_idx = df['total_time'].idxmax()
    min_idx = df['total_time'].idxmin()
    bars[max_idx].set_color('#E17055')
    bars[min_idx].set_color('#00B894')
    
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Total Execution Time (sec)')
    ax2.set_title('Total Execution Time of each Rank')
    ax2.grid(True, alpha=0.3)
    
    # 平均線を追加
    mean_time = df['total_time'].mean()
    ax2.axhline(y=mean_time, color='red', linestyle='--', alpha=0.7, label=f'average: {mean_time:.4f}s')
    ax2.legend()
    
    # 3. フェーズ別平均時間（円グラフ）
    ax3 = plt.subplot(subplot_layout[0], subplot_layout[1], 3)
    
    phase_means = [df[phase].mean() for phase in phases]
    wedges, texts, autotexts = ax3.pie(phase_means, labels=phase_labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax3.set_title('Percentage of average time by phase')
    
    # 4. 時間のボックスプロット
    ax4 = plt.subplot(subplot_layout[0], subplot_layout[1], 4)
    
    data_for_box = [df[phase] for phase in phases]
    box_plot = ax4.boxplot(data_for_box, labels=phase_labels, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('time (sec)')
    ax4.set_title('Phase-specific time distribution')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. 負荷バランス分析
    ax5 = plt.subplot(subplot_layout[0], subplot_layout[1], 5)
    
    # 各フェーズでの最大時間と最小時間の差
    load_imbalance = []
    for phase in phases:
        max_time = df[phase].max()
        min_time = df[phase].min()
        if df[phase].mean() > 0:
            imbalance = (max_time - min_time) / df[phase].mean() * 100
        else:
            imbalance = 0
        load_imbalance.append(imbalance)
    
    bars = ax5.bar(phase_labels, load_imbalance, color=colors, alpha=0.8)
    ax5.set_ylabel('load imbalance (%)')
    ax5.set_title('Phase-specific load imbalance')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 閾値線（20%）を追加
    ax5.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='attention level (20%)')
    ax5.legend()
    
    # 6. タイムライン表示（真のタイムライン）
    if has_timestamps:
        ax6 = plt.subplot(subplot_layout[0], subplot_layout[1], 6)
        
        # 真のタイムライン表示
        phase_start_cols = ['data_dist_start', 'host_to_dev_start', 'kernel_start', 
                           'dev_to_host_start', 'data_coll_start']
        phase_end_cols = ['data_dist_end', 'host_to_dev_end', 'kernel_end', 
                         'dev_to_host_end', 'data_coll_end']
        
        for rank in df['rank']:
            row = df[df['rank'] == rank].iloc[0]
            
            for i, (start_col, end_col, color, label) in enumerate(zip(phase_start_cols, phase_end_cols, colors, phase_labels)):
                start_time = row[start_col]
                end_time = row[end_col]
                duration = end_time - start_time
                
                ax6.barh(rank, duration, left=start_time, 
                        color=color, alpha=0.8, height=0.6)
        
        ax6.set_xlabel('Time from start (sec)')
        ax6.set_ylabel('Rank')
        ax6.set_title('True Timeline (Actual Start/End Times)')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-0.5, len(df) - 0.5)
        
        # 全体の実行時間範囲を表示
        global_start_min = df['global_start'].min()
        global_end_max = df['global_end'].max()
        ax6.set_xlim(global_start_min - 0.01, global_end_max + 0.01)
        
    else:
        # タイムスタンプデータがない場合は従来のタイムライン（積み上げ）
        ax6 = plt.subplot(subplot_layout[0], subplot_layout[1], 6)
        
        for rank in df['rank']:
            row = df[df['rank'] == rank].iloc[0]
            start_times = [0]
            
            for phase in phases:
                start_times.append(start_times[-1] + row[phase])
            
            for i, (phase, color) in enumerate(zip(phases, colors)):
                ax6.barh(rank, row[phase], left=start_times[i], 
                        color=color, alpha=0.8, height=0.6)
        
        ax6.set_xlabel('Cumulative time (sec)')
        ax6.set_ylabel('Rank')
        ax6.set_title('Cumulative Timeline (Sequential)')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-0.5, len(df) - 0.5)
    
    # レイアウト調整
    plt.tight_layout()
    
    # ファイル名から出力ファイル名を生成
    base_name = os.path.splitext(csv_file)[0]
    output_file = f"{base_name}_visualization.png"
    
    # 保存
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n可視化結果を保存しました: {output_file}")
    
    # 分析レポートを出力
    generate_analysis_report(df, phases, phase_labels, f"{base_name}_analysis.txt", has_timestamps)
    
    # 表示
    plt.show()

def generate_analysis_report(df, phases, phase_labels, output_file, has_timestamps=False):
    """
    分析レポートをテキストファイルに出力
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("マルチノード分散計算 タイミング分析レポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"タイムスタンプデータ: {'有り' if has_timestamps else '無し'}\n\n")
        
        # 基本統計
        f.write("【基本統計】\n")
        f.write(f"使用ランク数: {len(df)}\n")
        f.write(f"最大総実行時間: {df['total_time'].max():.6f} 秒 (ランク {df.loc[df['total_time'].idxmax(), 'rank']})\n")
        f.write(f"最小総実行時間: {df['total_time'].min():.6f} 秒 (ランク {df.loc[df['total_time'].idxmin(), 'rank']})\n")
        f.write(f"平均総実行時間: {df['total_time'].mean():.6f} 秒\n")
        f.write(f"標準偏差: {df['total_time'].std():.6f} 秒\n")
        f.write(f"変動係数: {(df['total_time'].std() / df['total_time'].mean()) * 100:.2f}%\n\n")
        
        # フェーズ別分析
        f.write("【フェーズ別分析】\n")
        for phase, label in zip(phases, phase_labels):
            mean_time = df[phase].mean()
            max_time = df[phase].max()
            min_time = df[phase].min()
            std_time = df[phase].std()
            percentage = (mean_time / df['total_time'].mean()) * 100
            
            f.write(f"\n{label}:\n")
            f.write(f"  平均時間: {mean_time:.6f} 秒 ({percentage:.1f}%)\n")
            f.write(f"  最大時間: {max_time:.6f} 秒 (ランク {df.loc[df[phase].idxmax(), 'rank']})\n")
            f.write(f"  最小時間: {min_time:.6f} 秒 (ランク {df.loc[df[phase].idxmin(), 'rank']})\n")
            f.write(f"  標準偏差: {std_time:.6f} 秒\n")
            if mean_time > 0:
                f.write(f"  負荷不均衡度: {((max_time - min_time) / mean_time) * 100:.2f}%\n")
            else:
                f.write(f"  負荷不均衡度: N/A (平均時間が0)\n")
        
        # タイムスタンプ分析
        if has_timestamps:
            f.write("\n【タイムスタンプ分析】\n")
            global_start_min = df['global_start'].min()
            global_start_max = df['global_start'].max()
            global_end_min = df['global_end'].min()
            global_end_max = df['global_end'].max()
            
            f.write(f"グローバル開始時刻の幅: {global_start_max - global_start_min:.6f} 秒\n")
            f.write(f"グローバル終了時刻の幅: {global_end_max - global_end_min:.6f} 秒\n")
            f.write(f"全体実行時間: {global_end_max - global_start_min:.6f} 秒\n")
            
            # 各フェーズの実際の重複時間を分析
            f.write("\n各フェーズの同期状況:\n")
            
            phase_start_cols = ['data_dist_start', 'host_to_dev_start', 'kernel_start', 
                               'dev_to_host_start', 'data_coll_start']
            phase_end_cols = ['data_dist_end', 'host_to_dev_end', 'kernel_end', 
                             'dev_to_host_end', 'data_coll_end']
            
            for i, (start_col, end_col, label) in enumerate(zip(phase_start_cols, phase_end_cols, phase_labels)):
                start_times = df[start_col]
                end_times = df[end_col]
                overlap_start = start_times.max()
                overlap_end = end_times.min()
                
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / (end_times.max() - start_times.min()) * 100
                    f.write(f"  {label}: {overlap_ratio:.1f}% 重複実行\n")
                else:
                    f.write(f"  {label}: 重複なし（完全に順次実行）\n")
        
        # ボトルネック分析
        f.write("\n【ボトルネック分析】\n")
        phase_means = [df[phase].mean() for phase in phases]
        bottleneck_idx = np.argmax(phase_means)
        bottleneck_phase = phase_labels[bottleneck_idx]
        bottleneck_time = phase_means[bottleneck_idx]
        bottleneck_percentage = (bottleneck_time / df['total_time'].mean()) * 100
        
        f.write(f"主要ボトルネック: {bottleneck_phase}\n")
        f.write(f"ボトルネック時間: {bottleneck_time:.6f} 秒 ({bottleneck_percentage:.1f}%)\n")
        
        # 改善提案
        f.write("\n【改善提案】\n")
        if bottleneck_percentage > 40:
            f.write(f"- {bottleneck_phase}が全体の40%以上を占めています。この部分の最適化を検討してください。\n")
        
        total_communication = df['data_distribution'].mean() + df['data_collection'].mean()
        comm_percentage = (total_communication / df['total_time'].mean()) * 100
        
        if comm_percentage > 30:
            f.write(f"- 通信時間が全体の{comm_percentage:.1f}%を占めています。データサイズや通信方法の最適化を検討してください。\n")
        
        load_imbalance_total = (df['total_time'].std() / df['total_time'].mean()) * 100
        if load_imbalance_total > 10:
            f.write(f"- 負荷不均衡度が{load_imbalance_total:.1f}%です。ワークロードの分散方法を見直してください。\n")
        
        if has_timestamps:
            # 実際のタイムライン分析に基づく改善提案
            phase_start_cols = ['data_dist_start', 'host_to_dev_start', 'kernel_start', 
                               'dev_to_host_start', 'data_coll_start']
            phase_end_cols = ['data_dist_end', 'host_to_dev_end', 'kernel_end', 
                             'dev_to_host_end', 'data_coll_end']
            
            total_overlap = 0
            for start_col, end_col in zip(phase_start_cols, phase_end_cols):
                start_times = df[start_col]
                end_times = df[end_col]
                overlap_start = start_times.max()
                overlap_end = end_times.min()
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start
            
            total_sequential_time = sum(df[phase].mean() for phase in phases)
            if total_overlap / total_sequential_time < 0.5:
                f.write("- 各フェーズの並列度が低いです。非同期処理や重複実行の最適化を検討してください。\n")
        
        f.write("\nレポート終了\n")
    
    print(f"分析レポートを保存しました: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python3 visualize_timing.py <csv_file>")
        print("例: python3 visualize_timing.py result/timing_data_20250716120000.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    create_timing_visualization(csv_file)