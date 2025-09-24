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

def load_demand_data():
    """加载需求数据以获取准确的时间信息"""
    try:
        demand_df = pd.read_csv('./data/demand.csv')
        time_df = demand_df.iloc[:, 0]  # 时间列
        # 只取第一天的数据（288个时间点）
        first_day_times = time_df.iloc[:288]
        return first_day_times.tolist()
    except Exception as e:
        print(f"加载需求数据失败: {e}")
        return []

def convert_time_slot_to_actual_time(time_slot, actual_times):
    """将时间段索引转换为实际时间"""
    if time_slot < len(actual_times):
        # 从 "2021-10-01T00:00" 格式提取时间部分
        time_str = actual_times[time_slot]
        if 'T' in time_str:
            return time_str.split('T')[1]
        return time_str
    else:
        # 如果超出范围，使用计算方式
        hours = (time_slot * 5) // 60
        minutes = (time_slot * 5) % 60
        return f"{hours:02d}:{minutes:02d}"

def analyze_first_day_distribution(free_slots_dic, actual_times):
    """分析第一天（288个时间段）的免费时间段分布"""
    # 只分析第一天的数据（时间段0-287）
    first_day_slots = {}
    total_free_slots = 0
    
    for time_slot, sites in free_slots_dic.items():
        if time_slot < 288:  # 只考虑第一天的数据
            first_day_slots[time_slot] = len(sites)
            total_free_slots += len(sites)
    
    # 按时间段排序
    sorted_time_slots = sorted(first_day_slots.items())
    
    return sorted_time_slots, total_free_slots

def create_detailed_time_distribution_plot(sorted_time_slots, actual_times):
    """创建详细的第一天时间分布图"""
    plt.figure(figsize=(20, 12))
    
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 创建实际时间标签
    time_labels = [convert_time_slot_to_actual_time(t, actual_times) for t in time_slots]
    
    # 主图：完整的第一天分布
    plt.subplot(3, 1, 1)
    bars = plt.bar(range(len(time_slots)), counts, alpha=0.8, color='skyblue', 
                   edgecolor='navy', linewidth=0.3, width=0.8)
    
    plt.title('第一天免费时间段分布 (2021-10-01 00:00-23:55)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('免费站点数量', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 设置x轴标签，每小时显示一个标签
    hour_indices = []
    hour_labels = []
    for i, time_slot in enumerate(time_slots):
        if time_slot % 12 == 0:  # 每12个时间段（1小时）显示一个标签
            hour_indices.append(i)
            hour_labels.append(convert_time_slot_to_actual_time(time_slot, actual_times))
    
    plt.xticks(hour_indices, hour_labels, rotation=45, ha='right')
    
    # 添加统计信息
    max_count = max(counts)
    max_index = counts.index(max_count)
    max_time_slot = time_slots[max_index]
    max_time_str = convert_time_slot_to_actual_time(max_time_slot, actual_times)
    
    plt.axvline(x=max_index, color='red', linestyle='--', alpha=0.7, linewidth=2,
                label=f'最高峰: {max_time_str} ({max_count}个免费站点)')
    plt.legend(fontsize=10)
    
    # 子图2：按小时聚合的分布
    plt.subplot(3, 1, 2)
    hourly_counts = defaultdict(list)
    
    for time_slot, count in sorted_time_slots:
        hour = time_slot // 12  # 每12个时间段为1小时
        hourly_counts[hour].append(count)
    
    hours = sorted(hourly_counts.keys())
    hourly_avg = [np.mean(hourly_counts[h]) for h in hours]
    hourly_max = [np.max(hourly_counts[h]) for h in hours]
    hourly_min = [np.min(hourly_counts[h]) for h in hours]
    
    x_pos = np.arange(len(hours))
    
    # 绘制平均值、最大值、最小值
    plt.plot(x_pos, hourly_avg, 'o-', color='blue', linewidth=2, markersize=6, label='平均值')
    plt.fill_between(x_pos, hourly_min, hourly_max, alpha=0.3, color='lightblue', label='最大-最小范围')
    
    plt.title('按小时聚合的免费站点数量分布', fontsize=14, fontweight='bold')
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('免费站点数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置x轴标签
    plt.xticks(x_pos, [f"{h:02d}:00" for h in hours])
    
    # 子图3：热力图风格的精细时间分布
    plt.subplot(3, 1, 3)
    
    # 创建24x12的矩阵（24小时 x 每小时12个5分钟时间段）
    heatmap_data = np.zeros((24, 12))
    
    for time_slot, count in sorted_time_slots:
        hour = time_slot // 12
        minute_slot = time_slot % 12
        if hour < 24:
            heatmap_data[hour, minute_slot] = count
    
    # 创建热力图
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='免费站点数量', shrink=0.8)
    
    plt.title('第一天24小时免费时间段热力图 (5分钟颗粒度)', fontsize=14, fontweight='bold')
    plt.xlabel('小时内的5分钟时间段', fontsize=12)
    plt.ylabel('小时', fontsize=12)
    
    # 设置刻度标签
    plt.yticks(range(24), [f"{i:02d}:00" for i in range(24)])
    plt.xticks(range(0, 12, 2), [f"{i*5:02d}分" for i in range(0, 12, 2)])
    
    # 在热力图上标注数值（只标注非零值）
    for hour in range(24):
        for minute_slot in range(12):
            value = heatmap_data[hour, minute_slot]
            if value > 0:
                plt.text(minute_slot, hour, f'{int(value)}', 
                        ha='center', va='center', fontsize=8, 
                        color='white' if value > np.max(heatmap_data) * 0.6 else 'black')
    
    plt.tight_layout()
    plt.savefig('data_analysis/first_day_free_slots_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_peak_periods_analysis(sorted_time_slots, actual_times):
    """创建高峰时段详细分析"""
    plt.figure(figsize=(18, 10))
    
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 计算高峰阈值
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    # 主图：时间序列图
    plt.subplot(2, 1, 1)
    
    # 绘制所有时间段
    x_indices = range(len(time_slots))
    plt.plot(x_indices, counts, color='lightblue', linewidth=1.5, alpha=0.8, label='免费站点数量')
    
    # 高亮显示高峰时段
    if peak_periods:
        peak_indices = [time_slots.index(t) for t, c in peak_periods]
        peak_counts = [c for t, c in peak_periods]
        plt.scatter(peak_indices, peak_counts, color='red', s=50, alpha=0.9, 
                   label=f'高峰时段 (>{threshold:.1f})', zorder=5)
    
    plt.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                label=f'高峰阈值: {threshold:.1f}')
    plt.axhline(y=mean_count, color='green', linestyle=':', alpha=0.7, linewidth=2,
                label=f'平均值: {mean_count:.1f}')
    
    plt.title('第一天免费时间段高峰分析', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('免费站点数量', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置x轴标签
    hour_indices = []
    hour_labels = []
    for i, time_slot in enumerate(time_slots):
        if time_slot % 24 == 0:  # 每2小时显示一个标签
            hour_indices.append(i)
            hour_labels.append(convert_time_slot_to_actual_time(time_slot, actual_times))
    
    plt.xticks(hour_indices, hour_labels, rotation=45, ha='right')
    
    # 子图：高峰时段分布统计
    plt.subplot(2, 1, 2)
    
    if peak_periods:
        # 按小时统计高峰时段
        hourly_peaks = defaultdict(int)
        peak_details = []
        
        for time_slot, count in peak_periods:
            hour = time_slot // 12
            hourly_peaks[hour] += 1
            actual_time = convert_time_slot_to_actual_time(time_slot, actual_times)
            peak_details.append((actual_time, count))
        
        hours = list(range(24))
        peak_counts_by_hour = [hourly_peaks[h] for h in hours]
        
        bars = plt.bar(hours, peak_counts_by_hour, alpha=0.8, color='orange', 
                      edgecolor='darkorange', linewidth=1)
        
        plt.title('各小时高峰时段数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('小时', fontsize=12)
        plt.ylabel('高峰时段数量', fontsize=12)
        plt.xticks(range(0, 24, 2), [f"{i:02d}:00" for i in range(0, 24, 2)])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 标注最高峰小时
        if peak_counts_by_hour and max(peak_counts_by_hour) > 0:
            max_hour = peak_counts_by_hour.index(max(peak_counts_by_hour))
            plt.text(max_hour, max(peak_counts_by_hour) + 0.1, 
                    f'最高峰\n{max_hour:02d}:00\n({max(peak_counts_by_hour)}个)', 
                    ha='center', va='bottom', fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 在条形图上标注数值
        for i, count in enumerate(peak_counts_by_hour):
            if count > 0:
                plt.text(i, count + 0.05, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data_analysis/first_day_peak_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_first_day_report(sorted_time_slots, actual_times, total_free_slots):
    """生成第一天分析报告"""
    report = []
    report.append("=" * 70)
    report.append("EADMM 第一天免费时间段分布详细分析报告")
    report.append("=" * 70)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"分析日期: 2021-10-01 (第一天数据)")
    report.append(f"时间范围: 00:00 - 23:55 (5分钟间隔)")
    report.append("")
    
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 基本统计信息
    report.append("1. 基本统计信息")
    report.append("-" * 40)
    report.append(f"总时间段数: 288 (第一天)")
    report.append(f"有免费站点的时间段数: {len([c for c in counts if c > 0])}")
    report.append(f"无免费站点的时间段数: {len([c for c in counts if c == 0])}")
    report.append(f"总免费时间段数: {total_free_slots}")
    report.append(f"平均每个时间段免费站点数: {np.mean(counts):.2f}")
    report.append(f"最大免费站点数: {max(counts)}")
    report.append(f"最小免费站点数: {min(counts)}")
    report.append("")
    
    # 时间分布分析
    report.append("2. 详细时间分布分析")
    report.append("-" * 40)
    
    # 找出最高峰时段
    max_count = max(counts)
    max_indices = [i for i, c in enumerate(counts) if c == max_count]
    
    report.append(f"最高峰免费站点数: {max_count}")
    report.append("最高峰时间段:")
    for idx in max_indices:
        time_slot = time_slots[idx]
        actual_time = convert_time_slot_to_actual_time(time_slot, actual_times)
        report.append(f"  - {actual_time} (时间段{time_slot})")
    
    # 前10个高峰时段
    top_10_peaks = sorted(sorted_time_slots, key=lambda x: x[1], reverse=True)[:10]
    report.append("\n前10个高峰时段:")
    for i, (time_slot, count) in enumerate(top_10_peaks, 1):
        actual_time = convert_time_slot_to_actual_time(time_slot, actual_times)
        report.append(f"  {i:2d}. {actual_time} - {count}个免费站点")
    
    # 按时间段分析（更细致的划分）
    report.append("\n按时间段详细分析:")
    
    time_periods = [
        ("深夜", 0, 6*12),      # 00:00-06:00
        ("清晨", 6*12, 9*12),   # 06:00-09:00  
        ("上午", 9*12, 12*12),  # 09:00-12:00
        ("中午", 12*12, 14*12), # 12:00-14:00
        ("下午", 14*12, 18*12), # 14:00-18:00
        ("傍晚", 18*12, 21*12), # 18:00-21:00
        ("晚间", 21*12, 24*12), # 21:00-24:00
    ]
    
    for period_name, start_slot, end_slot in time_periods:
        period_counts = [c for t, c in sorted_time_slots if start_slot <= t < end_slot]
        if period_counts:
            start_time = convert_time_slot_to_actual_time(start_slot, actual_times)
            end_time = convert_time_slot_to_actual_time(end_slot-1, actual_times)
            report.append(f"  {period_name} ({start_time}-{end_time}): "
                         f"平均 {np.mean(period_counts):.2f}, "
                         f"最大 {max(period_counts)}, "
                         f"最小 {min(period_counts)} 个免费站点")
    
    # 高峰时段分析
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    report.append(f"\n3. 高峰时段分析")
    report.append("-" * 40)
    report.append(f"高峰阈值 (平均值 + 标准差): {threshold:.2f}")
    report.append(f"高峰时段数量: {len(peak_periods)}")
    report.append(f"高峰时段占比: {len(peak_periods)/288*100:.1f}%")
    
    if peak_periods:
        # 按小时统计高峰时段
        hourly_peaks = defaultdict(int)
        for time_slot, count in peak_periods:
            hour = time_slot // 12
            hourly_peaks[hour] += 1
        
        if hourly_peaks:
            max_peak_hour = max(hourly_peaks.items(), key=lambda x: x[1])
            report.append(f"高峰最集中的小时: {max_peak_hour[0]:02d}:00 ({max_peak_hour[1]}个高峰时段)")
            
            report.append("\n各小时高峰时段分布:")
            for hour in range(24):
                if hourly_peaks[hour] > 0:
                    report.append(f"  {hour:02d}:00-{hour:02d}:59: {hourly_peaks[hour]}个高峰时段")
    
    # 零免费站点时段分析
    zero_periods = [t for t, c in sorted_time_slots if c == 0]
    if zero_periods:
        report.append(f"\n4. 零免费站点时段分析")
        report.append("-" * 40)
        report.append(f"零免费站点时段数量: {len(zero_periods)}")
        report.append(f"零免费站点时段占比: {len(zero_periods)/288*100:.1f}%")
        
        # 找出连续的零免费站点时段
        consecutive_zeros = []
        current_start = None
        
        for i, time_slot in enumerate(zero_periods):
            if current_start is None:
                current_start = time_slot
            elif time_slot != zero_periods[i-1] + 1:
                # 连续序列中断
                consecutive_zeros.append((current_start, zero_periods[i-1]))
                current_start = time_slot
        
        if current_start is not None:
            consecutive_zeros.append((current_start, zero_periods[-1]))
        
        if consecutive_zeros:
            report.append("\n连续零免费站点时段:")
            for start, end in consecutive_zeros:
                start_time = convert_time_slot_to_actual_time(start, actual_times)
                end_time = convert_time_slot_to_actual_time(end, actual_times)
                duration = end - start + 1
                report.append(f"  {start_time} - {end_time} (持续{duration*5}分钟)")
    
    report.append("")
    report.append("5. 结论与建议")
    report.append("-" * 40)
    
    # 基于分析结果的结论
    if peak_periods:
        peak_hours = [((t // 12)) for t, c in peak_periods]
        if peak_hours:
            most_common_hour = Counter(peak_hours).most_common(1)[0][0]
            report.append(f"• 免费时间段主要集中在 {most_common_hour:02d}:00 左右")
    
    # 分析活跃和非活跃时段
    high_activity_periods = [t for t, c in sorted_time_slots if c > mean_count]
    low_activity_periods = [t for t, c in sorted_time_slots if c < mean_count]
    
    report.append(f"• {len(high_activity_periods)} 个时间段的免费站点数量高于平均水平")
    report.append(f"• {len(low_activity_periods)} 个时间段的免费站点数量低于平均水平")
    
    if zero_periods:
        report.append(f"• 存在 {len(zero_periods)} 个完全没有免费站点的时间段，需要关注资源配置")
    
    report.append("• 建议在高峰时段优化资源配置以提高系统效率")
    report.append("• 可以考虑在低峰时段增加免费时间段以平衡负载")
    report.append("• 第一天数据显示了明显的时间模式，可用于预测和优化")
    
    # 保存报告
    with open('data_analysis/first_day_free_slots_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    return report

def save_first_day_summary(sorted_time_slots, actual_times, total_free_slots):
    """保存第一天分析摘要到JSON文件"""
    time_slots = [x[0] for x in sorted_time_slots]
    counts = [x[1] for x in sorted_time_slots]
    
    # 计算高峰时段
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + std_count
    peak_periods = [(t, c) for t, c in sorted_time_slots if c > threshold]
    
    # 前10个高峰时段
    top_10_peaks = sorted(sorted_time_slots, key=lambda x: x[1], reverse=True)[:10]
    top_10_peaks_info = []
    for t, c in top_10_peaks:
        actual_time = convert_time_slot_to_actual_time(t, actual_times)
        top_10_peaks_info.append({
            "time_slot": int(t), 
            "actual_time": actual_time,
            "free_sites_count": int(c)
        })
    
    # 按小时统计
    hourly_stats = {}
    for hour in range(24):
        hour_counts = [c for t, c in sorted_time_slots if t // 12 == hour]
        if hour_counts:
            hourly_stats[f"{hour:02d}:00"] = {
                "average": float(np.mean(hour_counts)),
                "maximum": int(max(hour_counts)),
                "minimum": int(min(hour_counts)),
                "total_slots": len(hour_counts)
            }
    
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "analysis_date": "2021-10-01",
        "time_range": "00:00 - 23:55",
        "granularity": "5 minutes",
        "basic_statistics": {
            "total_time_slots": 288,
            "time_slots_with_free_sites": len([c for c in counts if c > 0]),
            "time_slots_without_free_sites": len([c for c in counts if c == 0]),
            "total_free_slots": int(total_free_slots),
            "average_free_sites_per_slot": float(np.mean(counts)),
            "max_free_sites": int(max(counts)),
            "min_free_sites": int(min(counts)),
            "std_free_sites": float(np.std(counts))
        },
        "peak_analysis": {
            "peak_threshold": float(threshold),
            "peak_periods_count": len(peak_periods),
            "peak_periods_percentage": float(len(peak_periods) / 288 * 100)
        },
        "hourly_statistics": hourly_stats,
        "top_10_peak_periods": top_10_peaks_info
    }
    
    with open('data_analysis/first_day_free_slots_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("第一天分析摘要已保存到 data_analysis/first_day_free_slots_summary.json")

def main():
    """主函数"""
    print("开始分析EADMM第一天免费时间段分布...")
    print("注意：EADMM算法只使用第一天的数据（288个时间段）")
    
    # 加载数据
    free_slots_dic = load_free_slots_data()
    if not free_slots_dic:
        print("无法加载免费时间段数据，程序退出")
        return
    
    actual_times = load_demand_data()
    if not actual_times:
        print("无法加载时间数据，使用默认时间格式")
        actual_times = []
    
    print(f"成功加载免费时间段数据")
    
    # 分析第一天的时间分布
    sorted_time_slots, total_free_slots = analyze_first_day_distribution(free_slots_dic, actual_times)
    print(f"第一天总免费时间段数: {total_free_slots}")
    print(f"第一天时间段数: {len(sorted_time_slots)}")
    
    # 创建详细可视化图表
    print("生成第一天详细时间分布图...")
    create_detailed_time_distribution_plot(sorted_time_slots, actual_times)
    
    print("生成第一天高峰分析图...")
    create_peak_periods_analysis(sorted_time_slots, actual_times)
    
    # 生成分析报告
    print("生成第一天分析报告...")
    generate_first_day_report(sorted_time_slots, actual_times, total_free_slots)
    
    # 保存分析摘要
    print("保存第一天分析摘要...")
    save_first_day_summary(sorted_time_slots, actual_times, total_free_slots)
    
    print("第一天分析完成！生成的文件:")
    print("- data_analysis/first_day_free_slots_detailed.png")
    print("- data_analysis/first_day_peak_analysis.png")
    print("- data_analysis/first_day_free_slots_report.txt")
    print("- data_analysis/first_day_free_slots_summary.json")

if __name__ == "__main__":
    main()