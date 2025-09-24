import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, data_dir="data", output_dir="data_analysis"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.demand_df = None
        self.qos_df = None
        self.bandwidth_df = None
        
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据文件...")
        
        # 加载需求数据
        self.demand_df = pd.read_csv(f"{self.data_dir}/demand.csv")
        self.demand_df['mtime'] = pd.to_datetime(self.demand_df['mtime'])
        
        # 加载QoS数据
        self.qos_df = pd.read_csv(f"{self.data_dir}/qos.csv")
        
        # 加载站点带宽数据
        self.bandwidth_df = pd.read_csv(f"{self.data_dir}/site_bandwidth.csv")
        
        print(f"需求数据: {self.demand_df.shape}")
        print(f"QoS数据: {self.qos_df.shape}")
        print(f"带宽数据: {self.bandwidth_df.shape}")
        
    def analyze_demand_structure(self):
        """分析需求数据结构"""
        print("\n=== 需求数据结构分析 ===")
        
        # 基本信息
        print(f"时间范围: {self.demand_df['mtime'].min()} 到 {self.demand_df['mtime'].max()}")
        print(f"数据点数量: {len(self.demand_df)}")
        print(f"客户数量: {len(self.demand_df.columns) - 1}")
        
        # 客户列表
        customers = [col for col in self.demand_df.columns if col != 'mtime']
        print(f"客户列表: {customers}")
        
        # 统计信息
        demand_stats = self.demand_df[customers].describe()
        print("\n需求统计信息:")
        print(demand_stats)
        
        return customers, demand_stats
    
    def plot_demand_curves(self, customers=None, sample_hours=24):
        """绘制需求曲线"""
        print(f"\n=== 绘制需求曲线 (前{sample_hours}小时) ===")
        
        if customers is None:
            customers = [col for col in self.demand_df.columns if col != 'mtime']
        
        # 取前sample_hours小时的数据
        sample_data = self.demand_df.head(sample_hours * 12)  # 每小时12个数据点(5分钟间隔)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('客户需求曲线分析', fontsize=16)
        
        # 1. 总体需求趋势
        ax1 = axes[0, 0]
        total_demand = sample_data[customers].sum(axis=1)
        ax1.plot(sample_data['mtime'], total_demand, linewidth=2, color='blue')
        ax1.set_title('总体需求趋势')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('总需求量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 前10个客户的需求曲线
        ax2 = axes[0, 1]
        top_customers = customers[:10]
        for customer in top_customers:
            ax2.plot(sample_data['mtime'], sample_data[customer], 
                    label=customer, alpha=0.7, linewidth=1)
        ax2.set_title('前10个客户需求曲线')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('需求量')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 需求分布直方图
        ax3 = axes[1, 0]
        all_demands = sample_data[customers].values.flatten()
        ax3.hist(all_demands, bins=50, alpha=0.7, color='green')
        ax3.set_title('需求分布直方图')
        ax3.set_xlabel('需求量')
        ax3.set_ylabel('频次')
        
        # 4. 客户平均需求排序
        ax4 = axes[1, 1]
        avg_demands = sample_data[customers].mean().sort_values(ascending=False)
        top_20 = avg_demands.head(20)
        ax4.bar(range(len(top_20)), top_20.values, color='orange')
        ax4.set_title('前20个客户平均需求')
        ax4.set_xlabel('客户排名')
        ax4.set_ylabel('平均需求量')
        ax4.set_xticks(range(len(top_20)))
        ax4.set_xticklabels(top_20.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/demand_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_demands
    
    def analyze_time_patterns(self):
        """分析时间模式"""
        print("\n=== 时间模式分析 ===")
        
        # 添加时间特征
        self.demand_df['hour'] = self.demand_df['mtime'].dt.hour
        self.demand_df['day_of_week'] = self.demand_df['mtime'].dt.dayofweek
        self.demand_df['date'] = self.demand_df['mtime'].dt.date
        
        customers = [col for col in self.demand_df.columns 
                    if col not in ['mtime', 'hour', 'day_of_week', 'date']]
        
        # 按小时统计平均需求
        hourly_demand = self.demand_df.groupby('hour')[customers].mean()
        total_hourly = hourly_demand.sum(axis=1)
        
        # 按星期统计平均需求
        daily_demand = self.demand_df.groupby('day_of_week')[customers].mean()
        total_daily = daily_demand.sum(axis=1)
        
        # 绘制时间模式图
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 小时模式
        axes[0].plot(total_hourly.index, total_hourly.values, 
                    marker='o', linewidth=2, markersize=6)
        axes[0].set_title('24小时需求模式')
        axes[0].set_xlabel('小时')
        axes[0].set_ylabel('平均总需求量')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(0, 24, 2))
        
        # 星期模式
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        axes[1].bar(range(7), total_daily.values, color='skyblue')
        axes[1].set_title('一周需求模式')
        axes[1].set_xlabel('星期')
        axes[1].set_ylabel('平均总需求量')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_names)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return hourly_demand, daily_demand
    
    def analyze_qos_data(self):
        """分析QoS数据"""
        print("\n=== QoS数据分析 ===")
        
        # 设置站点名称为索引
        qos_matrix = self.qos_df.set_index('site_name')
        
        print(f"QoS矩阵形状: {qos_matrix.shape}")
        print(f"QoS值范围: {qos_matrix.min().min()} - {qos_matrix.max().max()}")
        
        # 绘制QoS热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(qos_matrix, annot=False, cmap='RdYlBu_r', 
                   cbar_kws={'label': 'QoS值'})
        plt.title('站点间QoS热力图')
        plt.xlabel('目标客户')
        plt.ylabel('源站点')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/qos_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # QoS统计
        qos_stats = qos_matrix.describe()
        print("\nQoS统计信息:")
        print(qos_stats)
        
        return qos_matrix, qos_stats
    
    def analyze_bandwidth_capacity(self):
        """分析站点带宽容量"""
        print("\n=== 站点带宽分析 ===")
        
        print(f"站点数量: {len(self.bandwidth_df)}")
        print(f"带宽范围: {self.bandwidth_df['bandwidth'].min()} - {self.bandwidth_df['bandwidth'].max()}")
        
        # 带宽统计
        bandwidth_stats = self.bandwidth_df['bandwidth'].describe()
        print("\n带宽统计信息:")
        print(bandwidth_stats)
        
        # 绘制带宽分布
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 带宽分布直方图
        axes[0].hist(self.bandwidth_df['bandwidth'], bins=20, 
                    color='lightcoral', alpha=0.7)
        axes[0].set_title('站点带宽分布')
        axes[0].set_xlabel('带宽')
        axes[0].set_ylabel('站点数量')
        
        # 站点带宽排序
        sorted_bandwidth = self.bandwidth_df.sort_values('bandwidth', ascending=False)
        axes[1].bar(range(len(sorted_bandwidth)), sorted_bandwidth['bandwidth'], 
                   color='lightgreen')
        axes[1].set_title('站点带宽排序')
        axes[1].set_xlabel('站点排名')
        axes[1].set_ylabel('带宽')
        axes[1].set_xticks(range(0, len(sorted_bandwidth), 5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bandwidth_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return bandwidth_stats, sorted_bandwidth
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("\n=== 综合数据分析 ===")
        
        # 加载数据
        self.load_data()
        
        # 分析需求数据结构
        customers, demand_stats = self.analyze_demand_structure()
        
        # 绘制需求曲线
        avg_demands = self.plot_demand_curves(customers)
        
        # 分析时间模式
        hourly_demand, daily_demand = self.analyze_time_patterns()
        
        # 分析QoS数据
        qos_matrix, qos_stats = self.analyze_qos_data()
        
        # 分析带宽容量
        bandwidth_stats, sorted_bandwidth = self.analyze_bandwidth_capacity()
        
        # 生成分析报告
        self.generate_report(customers, demand_stats, avg_demands, 
                           hourly_demand, daily_demand, qos_stats, bandwidth_stats)
        
        return {
            'customers': customers,
            'demand_stats': demand_stats,
            'avg_demands': avg_demands,
            'hourly_demand': hourly_demand,
            'daily_demand': daily_demand,
            'qos_matrix': qos_matrix,
            'qos_stats': qos_stats,
            'bandwidth_stats': bandwidth_stats,
            'sorted_bandwidth': sorted_bandwidth
        }
    
    def generate_report(self, customers, demand_stats, avg_demands, 
                       hourly_demand, daily_demand, qos_stats, bandwidth_stats):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")
        
        report = f"""
# 云计算数据分析报告

## 1. 数据概览
- 客户数量: {len(customers)}
- 需求数据时间范围: {self.demand_df['mtime'].min()} 到 {self.demand_df['mtime'].max()}
- 数据点数量: {len(self.demand_df)}
- 站点数量: {len(self.bandwidth_df)}

## 2. 需求分析
### 2.1 需求统计
- 平均需求: {demand_stats.loc['mean'].mean():.2f}
- 需求标准差: {demand_stats.loc['std'].mean():.2f}
- 最大需求: {demand_stats.loc['max'].max():.2f}
- 最小需求: {demand_stats.loc['min'].min():.2f}

### 2.2 高需求客户 (前10名)
{avg_demands.head(10).to_string()}

### 2.3 需求时间模式
- 峰值时段: {hourly_demand.sum(axis=1).idxmax()}:00
- 低谷时段: {hourly_demand.sum(axis=1).idxmin()}:00
- 需求波动系数: {(hourly_demand.sum(axis=1).std() / hourly_demand.sum(axis=1).mean()):.3f}

## 3. QoS分析
- QoS平均值: {qos_stats.loc['mean'].mean():.2f}
- QoS标准差: {qos_stats.loc['std'].mean():.2f}
- 最佳QoS: {qos_stats.loc['min'].min():.2f}
- 最差QoS: {qos_stats.loc['max'].max():.2f}

## 4. 带宽容量分析
- 平均带宽: {bandwidth_stats['mean']:.2f}
- 带宽标准差: {bandwidth_stats['std']:.2f}
- 最大带宽: {bandwidth_stats['max']:.2f}
- 最小带宽: {bandwidth_stats['min']:.2f}

## 5. 关键发现
1. 需求具有明显的时间周期性，存在明显的峰谷差异
2. 不同客户的需求差异较大，需要差异化的服务策略
3. QoS值在站点间存在显著差异，影响服务质量
4. 站点带宽配置不均衡，可能存在资源配置优化空间

## 6. 建议
1. 根据需求时间模式优化资源调度策略
2. 对高需求客户提供专门的服务保障
3. 优化站点间的QoS配置
4. 考虑带宽资源的重新分配和优化
        """
        
        # 保存报告
        with open(f'{self.output_dir}/data_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析报告已保存到 {self.output_dir}/data_analysis_report.txt")
        print(report)

if __name__ == "__main__":
    # 创建分析器实例，指定正确的数据路径和输出路径
    analyzer = DataAnalyzer(data_dir="../data", output_dir=".")
    
    # 执行综合分析
    results = analyzer.comprehensive_analysis()
    
    print("\n数据分析完成！生成的文件:")
    print(f"- {analyzer.output_dir}/demand_analysis.png: 需求分析图表")
    print(f"- {analyzer.output_dir}/time_patterns.png: 时间模式分析")
    print(f"- {analyzer.output_dir}/qos_heatmap.png: QoS热力图")
    print(f"- {analyzer.output_dir}/bandwidth_analysis.png: 带宽分析图表")
    print(f"- {analyzer.output_dir}/data_analysis_report.txt: 综合分析报告")