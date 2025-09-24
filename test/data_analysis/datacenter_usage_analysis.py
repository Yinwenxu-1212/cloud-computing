import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_eadmm_solution(filepath):
    """加载EADMM求解结果"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def analyze_datacenter_usage(solution_data):
    """分析数据中心使用情况"""
    # 提取数据
    x_variables = np.array(solution_data['x_variables'])  # (I*J, T)
    b_variables = np.array(solution_data['b_variables'])  # (J,)
    site_names = solution_data['site_names']
    customer_names = solution_data['customer_names']
    capacity = np.array(solution_data['capacity'])
    is_link = np.array(solution_data['is_link'])  # (J, I)
    
    I, J, T = solution_data['I'], solution_data['J'], solution_data['T']
    
    # 重塑x变量为 (I, J, T) 形状
    x_reshaped = x_variables.reshape((I, J, T))
    
    # 计算每个数据中心的总流量 (J, T)
    datacenter_flows = np.sum(x_reshaped, axis=0)  # 对所有客户求和
    
    # 计算每个数据中心在所有时间点的总流量
    total_flows_per_datacenter = np.sum(datacenter_flows, axis=1)  # (J,)
    
    # 计算每个数据中心的平均流量
    avg_flows_per_datacenter = np.mean(datacenter_flows, axis=1)  # (J,)
    
    # 计算每个数据中心的最大流量
    max_flows_per_datacenter = np.max(datacenter_flows, axis=1)  # (J,)
    
    # 分析使用情况
    used_datacenters = total_flows_per_datacenter > 1e-6  # 考虑数值精度
    unused_datacenters = ~used_datacenters
    
    num_used = np.sum(used_datacenters)
    num_unused = np.sum(unused_datacenters)
    usage_ratio = num_used / J
    
    # 计算带宽使用情况
    bandwidth_usage = b_variables / capacity * 100  # 百分比
    
    # 创建分析结果
    analysis_results = {
        'total_datacenters': J,
        'used_datacenters': num_used,
        'unused_datacenters': num_unused,
        'usage_ratio': usage_ratio,
        'used_datacenter_names': [site_names[i] for i in range(J) if used_datacenters[i]],
        'unused_datacenter_names': [site_names[i] for i in range(J) if unused_datacenters[i]],
        'total_flows_per_datacenter': total_flows_per_datacenter,
        'avg_flows_per_datacenter': avg_flows_per_datacenter,
        'max_flows_per_datacenter': max_flows_per_datacenter,
        'bandwidth_usage': bandwidth_usage,
        'b_variables': b_variables,
        'capacity': capacity,
        'datacenter_flows': datacenter_flows,
        'used_mask': used_datacenters,
        'site_names': site_names
    }
    
    return analysis_results

def create_visualizations(analysis_results):
    """创建可视化图表"""
    
    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 数据中心使用比例饼图
    ax1 = plt.subplot(2, 3, 1)
    labels = ['已使用', '未使用']
    sizes = [analysis_results['used_datacenters'], analysis_results['unused_datacenters']]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title(f'数据中心使用情况\n(总计: {analysis_results["total_datacenters"]}个)', fontsize=14, fontweight='bold')
    
    # 2. 数据中心流量热图
    ax2 = plt.subplot(2, 3, 2)
    datacenter_flows = analysis_results['datacenter_flows']
    used_mask = analysis_results['used_mask']
    
    # 只显示使用的数据中心
    used_flows = datacenter_flows[used_mask]
    used_names = [analysis_results['site_names'][i] for i in range(len(used_mask)) if used_mask[i]]
    
    if len(used_flows) > 0:
        # 限制显示的数据中心数量以避免过于拥挤
        max_display = 20
        if len(used_flows) > max_display:
            # 按总流量排序，显示前20个
            sorted_indices = np.argsort(np.sum(used_flows, axis=1))[::-1][:max_display]
            used_flows = used_flows[sorted_indices]
            used_names = [used_names[i] for i in sorted_indices]
        
        im = plt.imshow(used_flows, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax2, label='流量')
        plt.title('已使用数据中心流量热图', fontsize=14, fontweight='bold')
        plt.xlabel('时间点')
        plt.ylabel('数据中心')
        
        # 设置y轴标签
        if len(used_names) <= 10:
            plt.yticks(range(len(used_names)), used_names, fontsize=8)
        else:
            plt.yticks(range(0, len(used_names), max(1, len(used_names)//10)), 
                      [used_names[i] for i in range(0, len(used_names), max(1, len(used_names)//10))], 
                      fontsize=8)
    
    # 3. 带宽使用情况柱状图
    ax3 = plt.subplot(2, 3, 3)
    used_indices = np.where(used_mask)[0]
    if len(used_indices) > 0:
        used_bandwidth = analysis_results['bandwidth_usage'][used_indices]
        used_b_vars = analysis_results['b_variables'][used_indices]
        used_capacity = analysis_results['capacity'][used_indices]
        
        # 按带宽使用率排序
        sorted_indices = np.argsort(used_bandwidth)[::-1]
        
        # 限制显示数量
        max_display = 15
        if len(sorted_indices) > max_display:
            sorted_indices = sorted_indices[:max_display]
        
        x_pos = np.arange(len(sorted_indices))
        plt.bar(x_pos, used_bandwidth[sorted_indices], color='skyblue', alpha=0.7)
        plt.title('已使用数据中心带宽使用率', fontsize=14, fontweight='bold')
        plt.xlabel('数据中心')
        plt.ylabel('带宽使用率 (%)')
        
        # 修复索引错误
        used_site_names = [analysis_results['site_names'][i] for i in used_indices]
        plt.xticks(x_pos, [used_site_names[i] for i in sorted_indices], rotation=45, ha='right', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 4. 流量分布直方图
    ax4 = plt.subplot(2, 3, 4)
    total_flows = analysis_results['total_flows_per_datacenter']
    used_total_flows = total_flows[used_mask]
    
    if len(used_total_flows) > 0:
        plt.hist(used_total_flows, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('已使用数据中心总流量分布', fontsize=14, fontweight='bold')
        plt.xlabel('总流量')
        plt.ylabel('数据中心数量')
        plt.grid(True, alpha=0.3)
    
    # 5. 时间序列流量图（前几个最繁忙的数据中心）
    ax5 = plt.subplot(2, 3, 5)
    if len(used_flows) > 0:
        # 选择前5个最繁忙的数据中心
        top_n = min(5, len(used_flows))
        top_indices = np.argsort(np.sum(used_flows, axis=1))[::-1][:top_n]
        
        # 修复索引错误
        used_site_names = [analysis_results['site_names'][i] for i in np.where(used_mask)[0]]
        
        for i, idx in enumerate(top_indices):
            plt.plot(used_flows[idx], label=f'{used_site_names[idx]}', linewidth=2)
        
        plt.title('最繁忙数据中心流量时间序列', fontsize=14, fontweight='bold')
        plt.xlabel('时间点')
        plt.ylabel('流量')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 6. 容量利用率散点图
    ax6 = plt.subplot(2, 3, 6)
    if len(used_indices) > 0:
        used_capacity = analysis_results['capacity'][used_indices]
        used_max_flows = analysis_results['max_flows_per_datacenter'][used_indices]
        utilization_ratio = used_max_flows / used_capacity * 100
        
        plt.scatter(used_capacity, utilization_ratio, alpha=0.6, s=60, color='coral')
        plt.title('数据中心容量 vs 利用率', fontsize=14, fontweight='bold')
        plt.xlabel('容量')
        plt.ylabel('最大利用率 (%)')
        plt.grid(True, alpha=0.3)
        
        # 添加参考线
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100%利用率')
        plt.legend()
    
    plt.tight_layout()
    return fig

def generate_report(analysis_results):
    """生成分析报告"""
    report = []
    report.append("=" * 60)
    report.append("EADMM算法数据中心使用情况分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 基本统计
    report.append("1. 基本统计信息")
    report.append("-" * 30)
    report.append(f"总数据中心数量: {analysis_results['total_datacenters']}")
    report.append(f"已使用数据中心: {analysis_results['used_datacenters']}")
    report.append(f"未使用数据中心: {analysis_results['unused_datacenters']}")
    report.append(f"使用率: {analysis_results['usage_ratio']:.2%}")
    report.append("")
    
    # 流量统计
    used_mask = analysis_results['used_mask']
    total_flows = analysis_results['total_flows_per_datacenter']
    used_flows = total_flows[used_mask]
    
    if len(used_flows) > 0:
        report.append("2. 流量统计")
        report.append("-" * 30)
        report.append(f"已使用数据中心总流量: {np.sum(used_flows):.2f}")
        report.append(f"平均每个数据中心流量: {np.mean(used_flows):.2f}")
        report.append(f"最大数据中心流量: {np.max(used_flows):.2f}")
        report.append(f"最小数据中心流量: {np.min(used_flows):.2f}")
        report.append("")
    
    # 带宽使用情况
    used_indices = np.where(used_mask)[0]
    if len(used_indices) > 0:
        used_bandwidth = analysis_results['bandwidth_usage'][used_indices]
        report.append("3. 带宽使用情况")
        report.append("-" * 30)
        report.append(f"平均带宽使用率: {np.mean(used_bandwidth):.2f}%")
        report.append(f"最大带宽使用率: {np.max(used_bandwidth):.2f}%")
        report.append(f"最小带宽使用率: {np.min(used_bandwidth):.2f}%")
        report.append("")
    
    # 前10个最繁忙的数据中心
    if len(used_flows) > 0:
        top_10_indices = np.argsort(used_flows)[::-1][:10]
        report.append("4. 前10个最繁忙的数据中心")
        report.append("-" * 30)
        for i, idx in enumerate(top_10_indices):
            original_idx = np.where(used_mask)[0][idx]
            site_name = analysis_results['site_names'][original_idx]
            flow = used_flows[idx]
            bandwidth_usage = analysis_results['bandwidth_usage'][original_idx]
            report.append(f"{i+1:2d}. {site_name}: 总流量={flow:.2f}, 带宽使用率={bandwidth_usage:.2f}%")
        report.append("")
    
    # 未使用的数据中心
    if analysis_results['unused_datacenters'] > 0:
        report.append("5. 未使用的数据中心")
        report.append("-" * 30)
        unused_names = analysis_results['unused_datacenter_names']
        for i, name in enumerate(unused_names[:20]):  # 只显示前20个
            report.append(f"{i+1:2d}. {name}")
        if len(unused_names) > 20:
            report.append(f"... 还有 {len(unused_names) - 20} 个未使用的数据中心")
        report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    try:
        # 加载EADMM求解结果
        solution_file = '../experiment_results/eadmm_solution_detailed.json'
        print("正在加载EADMM求解结果...")
        solution_data = load_eadmm_solution(solution_file)
        
        # 分析数据中心使用情况
        print("正在分析数据中心使用情况...")
        analysis_results = analyze_datacenter_usage(solution_data)
        
        # 生成可视化图表
        print("正在生成可视化图表...")
        fig = create_visualizations(analysis_results)
        
        # 保存图表
        plt.savefig('datacenter_usage_analysis.png', dpi=300, bbox_inches='tight')
        print("可视化图表已保存为: datacenter_usage_analysis.png")
        
        # 生成报告
        print("正在生成分析报告...")
        report = generate_report(analysis_results)
        
        # 保存报告
        with open('datacenter_usage_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("分析报告已保存为: datacenter_usage_report.txt")
        
        # 保存详细分析结果
        analysis_summary = {
            'total_datacenters': int(analysis_results['total_datacenters']),
            'used_datacenters': int(analysis_results['used_datacenters']),
            'unused_datacenters': int(analysis_results['unused_datacenters']),
            'usage_ratio': float(analysis_results['usage_ratio']),
            'used_datacenter_names': analysis_results['used_datacenter_names'],
            'unused_datacenter_names': analysis_results['unused_datacenter_names']
        }
        
        with open('datacenter_usage_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        print("分析摘要已保存为: datacenter_usage_summary.json")
        
        # 显示基本统计信息
        print("\n" + "="*50)
        print("数据中心使用情况分析结果")
        print("="*50)
        print(f"总数据中心数量: {analysis_results['total_datacenters']}")
        print(f"已使用数据中心: {analysis_results['used_datacenters']}")
        print(f"未使用数据中心: {analysis_results['unused_datacenters']}")
        print(f"使用率: {analysis_results['usage_ratio']:.2%}")
        
        plt.show()
        
    except FileNotFoundError:
        print("错误: 找不到EADMM求解结果文件。请先运行Eadmm_test.py生成求解结果。")
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()