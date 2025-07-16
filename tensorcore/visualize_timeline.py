import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import sys
import re
import os

def parse_performance_data(filename):
    """性能データをパースする"""
    total_compute_times = {}
    operation_times = {}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # VISUALIZATION_DATA_START と VISUALIZATION_DATA_END の間のデータを抽出
    match = re.search(r'=== VISUALIZATION_DATA_START ===(.*?)=== VISUALIZATION_DATA_END ===', 
                     content, re.DOTALL)
    
    if not match:
        print("可視化データが見つかりませんでした")
        return None, None
    
    data_section = match.group(1)
    
    for line in data_section.strip().split('\n'):
        if line.startswith('TOTAL_COMPUTE_TIME,'):
            parts = line.split(',')
            method = parts[1]
            time_us = float(parts[2])
            total_compute_times[method] = time_us
        elif line.startswith('OPERATION_TIME,'):
            parts = line.split(',')
            operation = parts[1]
            time_us = float(parts[2])
            operation_times[operation] = time_us
    
    return total_compute_times, operation_times

def create_performance_charts(total_times, operation_times, output_dir):
    """性能比較チャートを作成"""
    if not total_times or not operation_times:
        print("データが不足しています")
        return
    
    # フォントサイズとスタイルの設定
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    # 1. 総計算時間の棒グラフ
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    methods = ['CUDA_Core_Global', 'CUDA_Core_Shared', 'TensorCore_Global', 'TensorCore_Shared']
    method_labels = ['CUDA Core\n(Global)', 'CUDA Core\n(Shared)', 'Tensor Core\n(Global)', 'Tensor Core\n(Shared)']
    times = [total_times.get(method, 0) for method in methods]
    
    # 色の設定
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars1 = ax1.bar(method_labels, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 値をバーの上に表示
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.1f} μs', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('実行時間 (μs)', fontweight='bold')
    ax1.set_title('総計算時間の比較\n(1024×1024×1024 行列乗算)', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 速度向上率の注釈を追加
    if times[0] > 0:  # CUDA Core Global を基準とする
        baseline = times[0]
        speedups = [baseline / t if t > 0 else 0 for t in times]
        
        # 速度向上率を表示
        for i, (bar, speedup) in enumerate(zip(bars1, speedups)):
            if i > 0 and speedup > 0:  # 基準以外
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                        f'{speedup:.1f}x', ha='center', va='center', 
                        fontweight='bold', color='white', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_compute_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"総計算時間比較グラフを {output_dir}/total_compute_time_comparison.png に保存しました")
    
    # 2. 各操作の時間比較
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    
    # 操作を論理的な順序で整理
    operation_order = [
        'CUDA_Core_Memory_Compute',
        'CUDA_Core_Shared_Declare',
        'CUDA_Core_Global_to_Shared',
        'CUDA_Core_Shared_Compute',
        'TensorCore_Fragment_Declare',
        'TensorCore_Fragment_Init',
        'TensorCore_Global_to_Frag',
        'TensorCore_Frag_to_Global',
        'TensorCore_Global_to_Shared',
        'TensorCore_Shared_to_Frag',
        'TensorCore_MMA_Sync'
    ]
    
    # 日本語ラベル
    operation_labels = [
        'CUDA Core\nメモリ込み計算',
        'CUDA Core\n共有メモリ宣言',
        'CUDA Core\nGlobal→Shared',
        'CUDA Core\n共有メモリ計算',
        'Tensor Core\nフラグメント宣言',
        'Tensor Core\nフラグメント初期化',
        'Tensor Core\nGlobal→Fragment',
        'Tensor Core\nFragment→Global',
        'Tensor Core\nGlobal→Shared',
        'Tensor Core\nShared→Fragment',
        'Tensor Core\nMMA演算'
    ]
    
    # データを取得（存在しない操作は0にする）
    operation_values = [operation_times.get(op, 0) for op in operation_order]
    
    # カテゴリ別の色分け
    operation_colors = [
        '#FF6B6B',  # CUDA Core Memory Compute
        '#FF9999',  # CUDA Core Shared Declare
        '#FFB366',  # CUDA Core Global to Shared
        '#FF6B6B',  # CUDA Core Shared Compute
        '#4ECDC4',  # TensorCore Fragment Declare
        '#7ED7D1',  # TensorCore Fragment Init
        '#45B7D1',  # TensorCore Global to Frag
        '#6BC5E8',  # TensorCore Frag to Global
        '#96CEB4',  # TensorCore Global to Shared
        '#B8D9C8',  # TensorCore Shared to Frag
        '#FFEAA7'   # TensorCore MMA Sync
    ]
    
    bars2 = ax2.bar(range(len(operation_labels)), operation_values, 
                    color=operation_colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 値をバーの上に表示（有効な値のみ）
    for i, (bar, value) in enumerate(zip(bars2, operation_values)):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(operation_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('操作', fontweight='bold')
    ax2.set_ylabel('実行時間 (μs)', fontweight='bold')
    ax2.set_title('各操作の時間比較\n(16×16×16 タイル基準で正規化)', fontweight='bold', pad=20)
    ax2.set_xticks(range(len(operation_labels)))
    ax2.set_xticklabels(operation_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Y軸をログスケールに設定（値の範囲が大きい場合）
    max_val = max([v for v in operation_values if v > 0])
    min_val = min([v for v in operation_values if v > 0])
    if max_val / min_val > 100:
        ax2.set_yscale('log')
        ax2.set_ylabel('実行時間 (μs, log scale)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/operation_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"操作別時間比較グラフを {output_dir}/operation_time_comparison.png に保存しました")
    
    # 3. 統計情報の表示
    print_performance_statistics(total_times, operation_times)

def print_performance_statistics(total_times, operation_times):
    """性能統計情報を表示"""
    print("\n=== 詳細統計情報 ===")
    
    # 総計算時間の比較
    print("\n1. 総計算時間比較:")
    methods = ['CUDA_Core_Global', 'CUDA_Core_Shared', 'TensorCore_Global', 'TensorCore_Shared']
    method_names = ['CUDA Core (Global)', 'CUDA Core (Shared)', 'Tensor Core (Global)', 'Tensor Core (Shared)']
    
    baseline_time = total_times.get('CUDA_Core_Global', 0)
    
    for method, name in zip(methods, method_names):
        time = total_times.get(method, 0)
        if time > 0:
            speedup = baseline_time / time if baseline_time > 0 else 1.0
            print(f"  {name}: {time:.1f} μs (速度向上: {speedup:.2f}x)")
    
    # 操作別の分析
    print("\n2. 操作別分析:")
    
    # CUDA Core vs Tensor Core の計算性能比較
    cuda_compute = operation_times.get('CUDA_Core_Memory_Compute', 0)
    tc_mma = operation_times.get('TensorCore_MMA_Sync', 0)
    
    if cuda_compute > 0 and tc_mma > 0:
        compute_speedup = cuda_compute / tc_mma
        print(f"  計算性能: Tensor Core は CUDA Core より {compute_speedup:.1f}x 高速")
    
    # メモリアクセスのオーバーヘッド分析
    tc_global_load = operation_times.get('TensorCore_Global_to_Frag', 0)
    tc_shared_load = operation_times.get('TensorCore_Global_to_Shared', 0) + operation_times.get('TensorCore_Shared_to_Frag', 0)
    
    if tc_global_load > 0 and tc_shared_load > 0:
        load_overhead = tc_shared_load / tc_global_load
        print(f"  ロードオーバーヘッド: 共有メモリ経由は直接アクセスより {load_overhead:.1f}x 遅い")
    
    # フラグメント操作のオーバーヘッド
    frag_declare = operation_times.get('TensorCore_Fragment_Declare', 0)
    frag_init = operation_times.get('TensorCore_Fragment_Init', 0)
    frag_overhead = frag_declare + frag_init
    
    if frag_overhead > 0 and tc_mma > 0:
        frag_ratio = frag_overhead / tc_mma
        print(f"  フラグメント初期化オーバーヘッド: MMA演算時間の {frag_ratio:.1f}倍")
    
    print("\n3. 最適化の提案:")
    
    # TensorCore vs CUDA Core の総合比較
    tc_global_total = total_times.get('TensorCore_Global', 0)
    cuda_global_total = total_times.get('CUDA_Core_Global', 0)
    
    if tc_global_total > 0 and cuda_global_total > 0:
        overall_speedup = cuda_global_total / tc_global_total
        print(f"  - Tensor Core により {overall_speedup:.1f}x の速度向上が達成されています")
    
    # 共有メモリの効果
    tc_shared_total = total_times.get('TensorCore_Shared', 0)
    if tc_global_total > 0 and tc_shared_total > 0:
        shared_effect = tc_shared_total / tc_global_total
        if shared_effect > 1.1:
            print(f"  - 共有メモリ使用により約 {shared_effect:.1f}x の性能低下が見られます")
            print("    → この行列サイズでは直接アクセスが効率的です")
        elif shared_effect < 0.9:
            print(f"  - 共有メモリ使用により約 {1/shared_effect:.1f}x の速度向上が見られます")
            print("    → 共有メモリの使用が効果的です")
        else:
            print("  - 共有メモリの効果は限定的です")
    
    # 最適な手法の推奨
    fastest_method = min(total_times.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
    print(f"  - 最高性能: {fastest_method[0]} ({fastest_method[1]:.1f} μs)")

def main():
    """メイン関数"""
    # 引数の処理
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
    else:
        output_filename = "performance_output.txt"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "."
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"性能データファイル: {output_filename}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # ファイルの存在確認
    if not os.path.exists(output_filename):
        print(f"エラー: ファイル '{output_filename}' が見つかりません")
        print("使用方法: python visualize_performance.py [performance_output.txt] [output_directory]")
        return
    
    # データの解析
    total_times, operation_times = parse_performance_data(output_filename)
    
    if total_times is None or operation_times is None:
        print("データの解析に失敗しました")
        return
    
    print(f"\n解析されたデータ:")
    print(f"  総計算時間データ: {len(total_times)} 項目")
    print(f"  操作別時間データ: {len(operation_times)} 項目")
    
    # チャートの作成
    create_performance_charts(total_times, operation_times, output_dir)
    
    print(f"\n可視化完了！グラフは {output_dir} ディレクトリに保存されました。")

if __name__ == "__main__":
    main()