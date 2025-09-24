import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import defaultdict, Counter
import json
from datetime import datetime, timedelta
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_free_slots_data():
    """加载免费时间段数据"""
    try:
        # 读取免费时间段数据
        free_slots_str = str(np.loadtxt('./test_solution/free_slots_M8928.txt', dtype=str, delimiter=";"))
        free_slots_dic = ast.literal_eval(free_slots_str)
        return free_slots_dic
    except Exception as e:
        print(f"加载免费时间段数据失败: {e}")
        return {}

def analyze_time_distribution(free_slots_dic):
    """分析免费时间段的时间分布"""
    # 统计每个时间段的免费站点数量
    time_slot_counts = {}
    total_free_slots = 0
    
    for time_slot, sites in free_slots_dic.items():
        time_slot_counts[time_slot] = len(sites)
        total_free_slots += len(sites)
    
    # 按时间段排序
    sorted_time_slots = sorted(time_slot_counts.items())
    
    return sorted_time_slots, total_free_slots

def analyze_site_distribution(free_slots_dic):
    """分析站点的免费时间段分布"""
    site_free_counts = defaultdict(int)
    site_time_slots = defaultdict(list)
    
    for time_slot, sites in free_slots_dic.items():
        for site in sites:
            site_free_counts[site] += 1
            site_time_slots[site].append(time_slot)
    
    return dict(site_free_counts), dict(site_time_slots)

def convert_to_time_format(time_slot):
    """将时间段索引转换为实际时间（5分钟间隔）"""
    # 假设从00:00开始，每个时间段5分钟
    hours = (time_slot * 5) // 60
    minutes = (time_slot * 5) % 60
    return f"{hours:02d}:{minutes:02d}"

def create_time_distribution_plot(sorted_time_slots):
    """创建时间分布图"""
    plt.figure(figsize=(15, 8))
    
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 创建条形图
    plt.subplot(2, 1, 1)
    plt.bar(time_slots, counts, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    plt.title('免费时间段分布 - 每个时间段的免费站点数量', fontsize=14, fontweight='bold')
    plt.xlabel('时间段索引 (5分钟间隔)', fontsize=12)
    plt.ylabel('免费站点数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    max_count = max(counts)
    max_time_slot = time_slots[counts.index(max_count)]
    plt.axvline(x=max_time_slot, color='red', linestyle='--', alpha=0.7, 
                label=f'最高峰: 时间段{max_time_slot} ({convert_to_time_format(max_time_slot)})')
    plt.legend()
    
    # 创建热力图风格的时间分布
    plt.subplot(2, 1, 2)
    
    # 将一天分为24小时，每小时12个时间段（5分钟间隔）
    hourly_data = np.zeros((24, 12))
    
    for time_slot, count in sorted_time_slots:
        if time_slot < 288:  # 一天288个时间段
            hour = (time_slot * 5) // 60
            minute_slot = ((time_slot * 5) % 60) // 5
            if hour < 24 and minute_slot < 12:
                hourly_data[hour, minute_slot] = count
    
    # 创建热力图
    im = plt.imshow(hourly_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='免费站点数量')
    plt.title('24小时免费时间段热力图', fontsize=14, fontweight='bold')
    plt.xlabel('小时内的5分钟时间段 (0-11)', fontsize=12)
    plt.ylabel('小时 (0-23)', fontsize=12)
    
    # 设置刻度
    plt.yticks(range(0, 24, 2), [f"{i:02d}:00" for i in range(0, 24, 2)])
    plt.xticks(range(0, 12, 2), [f"{i*5:02d}" for i in range(0, 12, 2)])
    
    plt.tight_layout()
    plt.savefig('data_analysis/free_slots_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_site_distribution_plot(site_free_counts):
    """创建站点分布图"""
    plt.figure(figsize=(15, 10))
    
    # 按免费时间段数量排序
    sorted_sites = sorted(site_free_counts.items(), key=lambda x: x[1], reverse=True)
    sites = [x[0] for x in sorted_sites]
    counts = [x[1] for x in sorted_sites]
    
    # 创建条形图
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(sites)), counts, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=0.5)
    plt.title('各站点免费时间段数量分布', fontsize=14, fontweight='bold')
    plt.xlabel('站点索引', fontsize=12)
    plt.ylabel('免费时间段数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 标注前10个最活跃的站点
    for i in range(min(10, len(sites))):
        plt.text(i, counts[i] + max(counts) * 0.01, f'站点{sites[i]}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 创建分布直方图
    plt.subplot(2, 1, 2)
    plt.hist(counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen', linewidth=1)
    plt.title('站点免费时间段数量分布直方图', fontsize=14, fontweight='bold')
    plt.xlabel('免费时间段数量', fontsize=12)
    plt.ylabel('站点数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    plt.axvline(x=mean_count, color='red', linestyle='--', alpha=0.7, label=f'平均值: {mean_count:.1f}')
    plt.axvline(x=median_count, color='blue', linestyle='--', alpha=0.7, label=f'中位数: {median_count:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_analysis/free_slots_site_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_peak_analysis_plot(sorted_time_slots):
    """创建高峰时段分析图"""
    plt.figure(figsize=(15, 8))
    
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 找出高峰时段（免费站点数量 > 平均值 + 标准差）
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    plt.subplot(2, 1, 1)
    # 绘制所有时间段
    plt.plot(time_slots, counts, alpha=0.6, color='lightblue', linewidth=1, label='所有时间段')
    
    # 高亮显示高峰时段
    if peak_periods:
        peak_times = [x[0] for x in peak_periods]
        peak_counts = [x[1] for x in peak_periods]
        plt.scatter(peak_times, peak_counts, color='red', s=30, alpha=0.8, label='高峰时段', zorder=5)
    
    plt.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'高峰阈值: {threshold:.1f}')
    plt.title('免费时间段高峰分析', fontsize=14, fontweight='bold')
    plt.xlabel('时间段索引', fontsize=12)
    plt.ylabel('免费站点数量', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 按小时统计高峰时段
    plt.subplot(2, 1, 2)
    hourly_peaks = defaultdict(int)
    
    for time_slot, count in peak_periods:
        hour = (time_slot * 5) // 60
        if hour < 24:
            hourly_peaks[hour] += 1
    
    hours = list(range(24))
    peak_counts_by_hour = [hourly_peaks[h] for h in hours]
    
    bars = plt.bar(hours, peak_counts_by_hour, alpha=0.7, color='orange', edgecolor='darkorange')
    plt.title('各小时高峰时段数量分布', fontsize=14, fontweight='bold')
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('高峰时段数量', fontsize=12)
    plt.xticks(range(0, 24, 2), [f"{i:02d}:00" for i in range(0, 24, 2)])
    plt.grid(True, alpha=0.3)
    
    # 标注最高峰小时
    if peak_counts_by_hour:
        max_hour = peak_counts_by_hour.index(max(peak_counts_by_hour))
        plt.text(max_hour, max(peak_counts_by_hour) + 0.5, f'最高峰\n{max_hour:02d}:00', 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('data_analysis/free_slots_peak_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report(free_slots_dic, sorted_time_slots, site_free_counts, total_free_slots):
    """生成分析报告"""
    report = []
    report.append("=" * 60)
    report.append("EADMM 免费时间段分布分析报告")
    report.append("=" * 60)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 基本统计信息
    report.append("1. 基本统计信息")
    report.append("-" * 30)
    report.append(f"总时间段数: {len(free_slots_dic)}")
    report.append(f"有免费站点的时间段数: {len([t for t, sites in free_slots_dic.items() if sites])}")
    report.append(f"总免费时间段数: {total_free_slots}")
    report.append(f"平均每个时间段免费站点数: {total_free_slots / len(free_slots_dic):.2f}")
    report.append("")
    
    # 时间分布分析
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    report.append("2. 时间分布分析")
    report.append("-" * 30)
    max_count = max(counts)
    max_time_slot = time_slots[counts.index(max_count)]
    max_time_str = convert_to_time_format(max_time_slot)
    
    report.append(f"最高峰时间段: {max_time_slot} ({max_time_str})")
    report.append(f"最高峰免费站点数: {max_count}")
    
    # 找出前5个高峰时段
    top_5_peaks = sorted(sorted_time_slots, key=lambda x: x[1], reverse=True)[:5]
    report.append("\n前5个高峰时段:")
    for i, (time_slot, count) in enumerate(top_5_peaks, 1):
        time_str = convert_to_time_format(time_slot)
        report.append(f"  {i}. 时间段{time_slot} ({time_str}): {count}个免费站点")
    
    # 按时间段分析
    morning_slots = [c for t, c in sorted_time_slots if 6*12 <= t < 12*12]  # 6:00-12:00
    afternoon_slots = [c for t, c in sorted_time_slots if 12*12 <= t < 18*12]  # 12:00-18:00
    evening_slots = [c for t, c in sorted_time_slots if 18*12 <= t < 24*12]  # 18:00-24:00
    night_slots = [c for t, c in sorted_time_slots if t < 6*12 or t >= 24*12]  # 0:00-6:00
    
    report.append(f"\n时段分析:")
    report.append(f"  夜间 (00:00-06:00): 平均 {np.mean(night_slots) if night_slots else 0:.2f} 个免费站点")
    report.append(f"  上午 (06:00-12:00): 平均 {np.mean(morning_slots) if morning_slots else 0:.2f} 个免费站点")
    report.append(f"  下午 (12:00-18:00): 平均 {np.mean(afternoon_slots) if afternoon_slots else 0:.2f} 个免费站点")
    report.append(f"  晚间 (18:00-24:00): 平均 {np.mean(evening_slots) if evening_slots else 0:.2f} 个免费站点")
    report.append("")
    
    # 站点分布分析
    report.append("3. 站点分布分析")
    report.append("-" * 30)
    report.append(f"参与免费时段的站点总数: {len(site_free_counts)}")
    
    sorted_sites = sorted(site_free_counts.items(), key=lambda x: x[1], reverse=True)
    report.append(f"最活跃站点: 站点{sorted_sites[0][0]} ({sorted_sites[0][1]}个免费时间段)")
    
    # 前10个最活跃站点
    report.append("\n前10个最活跃站点:")
    for i, (site, count) in enumerate(sorted_sites[:10], 1):
        report.append(f"  {i}. 站点{site}: {count}个免费时间段")
    
    # 站点活跃度分布
    counts_list = list(site_free_counts.values())
    report.append(f"\n站点活跃度统计:")
    report.append(f"  平均免费时间段数: {np.mean(counts_list):.2f}")
    report.append(f"  中位数免费时间段数: {np.median(counts_list):.2f}")
    report.append(f"  标准差: {np.std(counts_list):.2f}")
    report.append("")
    
    # 高峰时段分析
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    report.append("4. 高峰时段分析")
    report.append("-" * 30)
    report.append(f"高峰阈值 (平均值 + 标准差): {threshold:.2f}")
    report.append(f"高峰时段数量: {len(peak_periods)}")
    
    if peak_periods:
        # 按小时统计高峰时段
        hourly_peaks = defaultdict(int)
        for time_slot, count in peak_periods:
            hour = (time_slot * 5) // 60
            if hour < 24:
                hourly_peaks[hour] += 1
        
        if hourly_peaks:
            max_peak_hour = max(hourly_peaks.items(), key=lambda x: x[1])
            report.append(f"高峰最集中的小时: {max_peak_hour[0]:02d}:00 ({max_peak_hour[1]}个高峰时段)")
    
    report.append("")
    report.append("5. 结论与建议")
    report.append("-" * 30)
    
    # 基于分析结果的结论
    if peak_periods:
        peak_hours = [((t * 5) // 60) for t, c in peak_periods if (t * 5) // 60 < 24]
        if peak_hours:
            most_common_hour = Counter(peak_hours).most_common(1)[0][0]
            report.append(f"• 免费时间段主要集中在 {most_common_hour:02d}:00 左右")
    
    if sorted_sites:
        active_sites = len([s for s, c in site_free_counts.items() if c > np.mean(counts_list)])
        report.append(f"• {active_sites} 个站点的免费时间段数量高于平均水平")
    
    report.append("• 建议在高峰时段优化资源配置以提高系统效率")
    report.append("• 可以考虑在低峰时段增加免费时间段以平衡负载")
    
    # 保存报告
    with open('data_analysis/free_slots_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    return report

def save_analysis_summary(free_slots_dic, sorted_time_slots, site_free_counts, total_free_slots):
    """保存分析摘要到JSON文件"""
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 计算高峰时段
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    # 前10个最活跃站点
    sorted_sites = sorted(site_free_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_sites = [{"site_id": int(site), "free_slots_count": int(count)} 
                    for site, count in sorted_sites[:10]]
    
    # 前5个高峰时段
    top_5_peaks = sorted(sorted_time_slots, key=lambda x: x[1], reverse=True)[:5]
    top_5_peaks_info = [{"time_slot": int(t), "time_str": convert_to_time_format(t), 
                        "free_sites_count": int(c)} for t, c in top_5_peaks]
    
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "basic_statistics": {
            "total_time_slots": len(free_slots_dic),
            "time_slots_with_free_sites": len([t for t, sites in free_slots_dic.items() if sites]),
            "total_free_slots": int(total_free_slots),
            "average_free_sites_per_slot": float(total_free_slots / len(free_slots_dic))
        },
        "peak_analysis": {
            "peak_threshold": float(threshold),
            "peak_periods_count": len(peak_periods),
            "max_free_sites": int(max(counts)),
            "max_free_sites_time_slot": int(time_slots[counts.index(max(counts))]),
            "max_free_sites_time_str": convert_to_time_format(time_slots[counts.index(max(counts))])
        },
        "site_statistics": {
            "total_participating_sites": len(site_free_counts),
            "average_free_slots_per_site": float(np.mean(list(site_free_counts.values()))),
            "median_free_slots_per_site": float(np.median(list(site_free_counts.values()))),
            "std_free_slots_per_site": float(np.std(list(site_free_counts.values())))
        },
        "top_10_most_active_sites": top_10_sites,
        "top_5_peak_periods": top_5_peaks_info
    }
    
    with open('data_analysis/free_slots_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("分析摘要已保存到 data_analysis/free_slots_analysis_summary.json")

def main():
    """主函数"""
    print("开始分析EADMM免费时间段分布...")
    
    # 加载数据
    free_slots_dic = load_free_slots_data()
    if not free_slots_dic:
        print("无法加载免费时间段数据，程序退出")
        return
    
    print(f"成功加载 {len(free_slots_dic)} 个时间段的数据")
    
    # 分析时间分布
    sorted_time_slots, total_free_slots = analyze_time_distribution(free_slots_dic)
    print(f"总免费时间段数: {total_free_slots}")
    
    # 分析站点分布
    site_free_counts, site_time_slots = analyze_site_distribution(free_slots_dic)
    print(f"参与免费时段的站点数: {len(site_free_counts)}")
    
    # 创建可视化图表
    print("生成时间分布图...")
    create_time_distribution_plot(sorted_time_slots)
    
    print("生成站点分布图...")
    create_site_distribution_plot(site_free_counts)
    
    print("生成高峰分析图...")
    create_peak_analysis_plot(sorted_time_slots)
    
    # 生成分析报告
    print("生成分析报告...")
    generate_analysis_report(free_slots_dic, sorted_time_slots, site_free_counts, total_free_slots)
    
    # 保存分析摘要
    print("保存分析摘要...")
    save_analysis_summary(free_slots_dic, sorted_time_slots, site_free_counts, total_free_slots)
    
    print("分析完成！生成的文件:")
    print("- data_analysis/free_slots_time_distribution.png")
    print("- data_analysis/free_slots_site_distribution.png") 
    print("- data_analysis/free_slots_peak_analysis.png")
    print("- data_analysis/free_slots_analysis_report.txt")
    print("- data_analysis/free_slots_analysis_summary.json")

if __name__ == "__main__":
    main()