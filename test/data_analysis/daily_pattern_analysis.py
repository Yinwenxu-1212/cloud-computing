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

class DailyPatternAnalyzer:
    def __init__(self, data_dir="../data", output_dir="."):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.demand_df = None
        
    def load_data(self):
        """加载需求数据"""
        print("正在加载需求数据...")
        self.demand_df = pd.read_csv(f"{self.data_dir}/demand.csv")
        self.demand_df['mtime'] = pd.to_datetime(self.demand_df['mtime'])
        
        # 添加时间特征
        self.demand_df['hour'] = self.demand_df['mtime'].dt.hour
        self.demand_df['minute'] = self.demand_df['mtime'].dt.minute
        self.demand_df['date'] = self.demand_df['mtime'].dt.date
        self.demand_df['day_of_month'] = self.demand_df['mtime'].dt.day
        self.demand_df['weekday'] = self.demand_df['mtime'].dt.dayofweek
        
        # 创建时间槽（5分钟间隔，一天288个时间槽）
        self.demand_df['time_slot'] = self.demand_df['hour'] * 12 + self.demand_df['minute'] // 5
        
        print(f"数据形状: {self.demand_df.shape}")
        print(f"时间范围: {self.demand_df['mtime'].min()} 到 {self.demand_df['mtime'].max()}")
        print(f"总天数: {len(self.demand_df['date'].unique())}")
        
    def analyze_daily_consistency(self):
        """分析每日需求模式的一致性"""
        print("\n=== 分析每日需求模式一致性 ===")
        
        customers = [col for col in self.demand_df.columns 
                    if col not in ['mtime', 'hour', 'minute', 'date', 'day_of_month', 'weekday', 'time_slot']]
        
        # 计算每日总需求
        daily_total_demand = self.demand_df.groupby('date')[customers].sum().sum(axis=1)
        
        # 按日期和时间槽分组，计算每个时间槽的总需求
        hourly_patterns = self.demand_df.groupby(['date', 'time_slot'])[customers].sum().sum(axis=1).reset_index()
        hourly_patterns.columns = ['date', 'time_slot', 'total_demand']
        
        # 转换为透视表，行为日期，列为时间槽
        daily_patterns_matrix = hourly_patterns.pivot(index='date', columns='time_slot', values='total_demand')
        
        # 计算每日模式的相关性
        correlation_matrix = daily_patterns_matrix.T.corr()
        
        return daily_patterns_matrix, correlation_matrix, daily_total_demand
    
    def plot_daily_patterns_comparison(self, daily_patterns_matrix, num_days=None):
        """绘制多日需求模式对比"""
        if num_days is None:
            num_days = len(daily_patterns_matrix)
            selected_days = daily_patterns_matrix
            title = f'所有{num_days}天需求模式对比'
            filename = 'all_daily_patterns_comparison.png'
        else:
            selected_days = daily_patterns_matrix.head(num_days)
            title = f'前{num_days}天需求模式对比'
            filename = 'daily_patterns_comparison.png'
        
        print(f"\n=== 绘制{title} ===")
        
        plt.figure(figsize=(20, 12))
        
        # 使用颜色映射来区分不同的天
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_days)))
        
        # 绘制每日模式
        for i, (date, pattern) in enumerate(selected_days.iterrows()):
            plt.plot(pattern.index, pattern.values, 
                    label=f'{date} (第{i+1}天)', 
                    linewidth=1.5, alpha=0.7, color=colors[i])
        
        plt.title(title, fontsize=18)
        plt.xlabel('时间槽 (5分钟间隔)', fontsize=14)
        plt.ylabel('总需求量', fontsize=14)
        
        # 如果天数太多，不显示图例
        if num_days is None or num_days > 15:
            plt.legend().set_visible(False)
            # 添加文本说明
            plt.text(0.02, 0.98, f'显示了{len(selected_days)}天的需求模式\n(图例已隐藏以保持清晰)', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, alpha=0.3)
        
        # 添加小时标记
        hour_ticks = [i*12 for i in range(0, 25, 2)]  # 每2小时一个标记
        hour_labels = [f'{i}:00' for i in range(0, 25, 2)]
        plt.xticks(hour_ticks, hour_labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_all_days_heatmap(self, daily_patterns_matrix):
        """绘制所有天数的需求模式热力图"""
        print("\n=== 绘制所有天数需求模式热力图 ===")
        
        # 标准化数据（按行标准化，每天的模式标准化到0-1）
        normalized_patterns = daily_patterns_matrix.div(daily_patterns_matrix.max(axis=1), axis=0)
        
        plt.figure(figsize=(20, 12))
        
        # 创建热力图
        sns.heatmap(normalized_patterns, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': '标准化需求强度'},
                   xticklabels=False,  # 时间槽太多，不显示标签
                   yticklabels=True)
        
        plt.title('所有天数的需求模式热力图（标准化）', fontsize=16)
        plt.xlabel('时间槽 (5分钟间隔)', fontsize=12)
        plt.ylabel('日期', fontsize=12)
        
        # 添加小时标记
        hour_positions = [i*12 for i in range(0, 25, 2)]  # 每2小时一个标记
        hour_labels = [f'{i}:00' for i in range(0, 25, 2)]
        plt.xticks(hour_positions, hour_labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_days_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_pattern_similarity(self, correlation_matrix):
        """分析模式相似性"""
        print("\n=== 分析模式相似性 ===")
        
        # 计算平均相关系数（排除对角线）
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_triangle = correlation_matrix.where(mask)
        avg_correlation = upper_triangle.mean().mean()
        
        print(f"日间需求模式平均相关系数: {avg_correlation:.4f}")
        
        # 找出相关性最高和最低的日期对
        correlation_values = upper_triangle.stack().dropna()
        max_corr_pair = correlation_values.idxmax()
        min_corr_pair = correlation_values.idxmin()
        
        print(f"最相似的两天: {max_corr_pair[0]} 和 {max_corr_pair[1]}, 相关系数: {correlation_values[max_corr_pair]:.4f}")
        print(f"最不相似的两天: {min_corr_pair[0]} 和 {min_corr_pair[1]}, 相关系数: {correlation_values[min_corr_pair]:.4f}")
        
        # 绘制相关性分布
        plt.figure(figsize=(12, 5))
        
        # 相关性直方图
        plt.subplot(1, 2, 1)
        plt.hist(correlation_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_correlation, color='red', linestyle='--', linewidth=2, label=f'平均值: {avg_correlation:.3f}')
        plt.title('日间需求模式相关性分布')
        plt.xlabel('相关系数')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 相关性热力图（部分）
        plt.subplot(1, 2, 2)
        # 只显示前15天的相关性矩阵，避免过于密集
        subset_corr = correlation_matrix.iloc[:15, :15]
        sns.heatmap(subset_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': '相关系数'})
        plt.title('前15天需求模式相关性矩阵')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pattern_similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_correlation, max_corr_pair, min_corr_pair
    
    def analyze_weekday_vs_weekend(self):
        """分析工作日与周末的模式差异"""
        print("\n=== 分析工作日与周末模式差异 ===")
        
        customers = [col for col in self.demand_df.columns 
                    if col not in ['mtime', 'hour', 'minute', 'date', 'day_of_month', 'weekday', 'time_slot']]
        
        # 区分工作日和周末
        self.demand_df['is_weekend'] = self.demand_df['weekday'].isin([5, 6])  # 周六、周日
        
        # 按工作日/周末和时间槽分组
        weekday_pattern = self.demand_df[~self.demand_df['is_weekend']].groupby('time_slot')[customers].sum().sum(axis=1)
        weekend_pattern = self.demand_df[self.demand_df['is_weekend']].groupby('time_slot')[customers].sum().sum(axis=1)
        
        # 标准化（按最大值标准化）
        weekday_pattern_norm = weekday_pattern / weekday_pattern.max()
        weekend_pattern_norm = weekend_pattern / weekend_pattern.max()
        
        plt.figure(figsize=(15, 6))
        
        plt.plot(weekday_pattern_norm.index, weekday_pattern_norm.values, 
                label='工作日模式', linewidth=3, color='blue')
        plt.plot(weekend_pattern_norm.index, weekend_pattern_norm.values, 
                label='周末模式', linewidth=3, color='red')
        
        plt.title('工作日与周末需求模式对比（标准化）', fontsize=16)
        plt.xlabel('时间槽 (5分钟间隔)', fontsize=12)
        plt.ylabel('标准化需求强度', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加小时标记
        hour_ticks = [i*12 for i in range(0, 25, 2)]
        hour_labels = [f'{i}:00' for i in range(0, 25, 2)]
        plt.xticks(hour_ticks, hour_labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算工作日和周末模式的相关性
        correlation = weekday_pattern_norm.corr(weekend_pattern_norm)
        print(f"工作日与周末模式相关系数: {correlation:.4f}")
        
        return weekday_pattern_norm, weekend_pattern_norm, correlation
    
    def generate_daily_pattern_report(self, avg_correlation, max_corr_pair, min_corr_pair, 
                                    weekday_weekend_corr, daily_total_demand):
        """生成每日模式分析报告"""
        print("\n=== 生成每日模式分析报告 ===")
        
        # 计算每日需求的变异系数
        daily_demand_cv = daily_total_demand.std() / daily_total_demand.mean()
        
        report = f"""
# 每日需求模式一致性分析报告

## 1. 数据概览
- 分析天数: {len(daily_total_demand)}天
- 时间范围: {self.demand_df['mtime'].min().strftime('%Y-%m-%d')} 到 {self.demand_df['mtime'].max().strftime('%Y-%m-%d')}
- 每日时间槽数: 288个 (5分钟间隔)

## 2. 每日需求总量分析
- 平均每日总需求: {daily_total_demand.mean():.2f}
- 每日需求标准差: {daily_total_demand.std():.2f}
- 每日需求变异系数: {daily_demand_cv:.4f}
- 最高需求日: {daily_total_demand.idxmax()} ({daily_total_demand.max():.2f})
- 最低需求日: {daily_total_demand.idxmin()} ({daily_total_demand.min():.2f})

## 3. 需求模式一致性分析
- **日间需求模式平均相关系数: {avg_correlation:.4f}**
- 最相似的两天: {max_corr_pair[0]} 和 {max_corr_pair[1]}
- 最不相似的两天: {min_corr_pair[0]} 和 {min_corr_pair[1]}

## 4. 工作日与周末对比
- **工作日与周末模式相关系数: {weekday_weekend_corr:.4f}**

## 5. 关键发现

### 5.1 模式一致性评估
"""
        
        if avg_correlation > 0.8:
            consistency_level = "非常高"
            pattern_desc = "各天的需求模式高度一致，存在稳定的日周期规律"
        elif avg_correlation > 0.6:
            consistency_level = "较高"
            pattern_desc = "各天的需求模式比较一致，但存在一定变化"
        elif avg_correlation > 0.4:
            consistency_level = "中等"
            pattern_desc = "各天的需求模式存在明显差异，但仍有一定规律性"
        else:
            consistency_level = "较低"
            pattern_desc = "各天的需求模式差异较大，规律性不强"
        
        report += f"- 模式一致性水平: **{consistency_level}** (相关系数: {avg_correlation:.4f})\n"
        report += f"- {pattern_desc}\n\n"
        
        if weekday_weekend_corr > 0.7:
            weekend_desc = "工作日与周末的需求模式相似"
        elif weekday_weekend_corr > 0.5:
            weekend_desc = "工作日与周末的需求模式有一定差异"
        else:
            weekend_desc = "工作日与周末的需求模式存在显著差异"
        
        report += f"### 5.2 工作日与周末差异\n"
        report += f"- {weekend_desc} (相关系数: {weekday_weekend_corr:.4f})\n\n"
        
        if daily_demand_cv < 0.1:
            demand_stability = "非常稳定"
        elif daily_demand_cv < 0.2:
            demand_stability = "比较稳定"
        elif daily_demand_cv < 0.3:
            demand_stability = "中等波动"
        else:
            demand_stability = "波动较大"
        
        report += f"### 5.3 需求总量稳定性\n"
        report += f"- 每日总需求{demand_stability} (变异系数: {daily_demand_cv:.4f})\n\n"
        
        report += f"""
## 6. 结论与建议

### 6.1 主要结论
1. **需求模式规律性**: 基于{avg_correlation:.4f}的平均相关系数，需求具有{'强' if avg_correlation > 0.7 else '中等' if avg_correlation > 0.5 else '弱'}规律性
2. **时间预测可行性**: {'高' if avg_correlation > 0.7 else '中等' if avg_correlation > 0.5 else '低'}
3. **调度策略适用性**: {'可以使用统一的调度策略' if avg_correlation > 0.7 else '需要考虑日间差异的调度策略'}

### 6.2 优化建议
1. **资源调度**: {'基于稳定的日周期模式进行资源预分配' if avg_correlation > 0.7 else '需要动态调整资源分配策略'}
2. **容量规划**: 考虑{daily_demand_cv:.1%}的日间需求变化进行容量规划
3. **预测模型**: {'可以使用基于历史模式的简单预测模型' if avg_correlation > 0.7 else '建议使用更复杂的机器学习预测模型'}

## 7. 生成文件
- all_daily_patterns_comparison.png: 所有{len(daily_total_demand)}天需求模式对比
- daily_patterns_comparison.png: 前7天需求模式对比
- all_days_heatmap.png: 所有天数需求模式热力图
- pattern_similarity_analysis.png: 模式相似性分析
- weekday_vs_weekend.png: 工作日与周末对比
- daily_pattern_report.txt: 本分析报告
        """
        
        # 保存报告
        with open(f'{self.output_dir}/daily_pattern_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"每日模式分析报告已保存到 {self.output_dir}/daily_pattern_report.txt")
        print(report)
    
    def comprehensive_daily_analysis(self):
        """执行完整的每日模式分析"""
        print("=== 开始每日需求模式一致性分析 ===")
        
        # 加载数据
        self.load_data()
        
        # 分析每日一致性
        daily_patterns_matrix, correlation_matrix, daily_total_demand = self.analyze_daily_consistency()
        
        # 绘制对比图（所有天数）
        self.plot_daily_patterns_comparison(daily_patterns_matrix)
        
        # 绘制对比图（前7天）
        self.plot_daily_patterns_comparison(daily_patterns_matrix, num_days=7)
        
        # 绘制热力图
        self.plot_all_days_heatmap(daily_patterns_matrix)
        
        # 分析相似性
        avg_correlation, max_corr_pair, min_corr_pair = self.analyze_pattern_similarity(correlation_matrix)
        
        # 分析工作日与周末
        weekday_pattern, weekend_pattern, weekday_weekend_corr = self.analyze_weekday_vs_weekend()
        
        # 生成报告
        self.generate_daily_pattern_report(avg_correlation, max_corr_pair, min_corr_pair, 
                                         weekday_weekend_corr, daily_total_demand)
        
        return {
            'daily_patterns_matrix': daily_patterns_matrix,
            'correlation_matrix': correlation_matrix,
            'avg_correlation': avg_correlation,
            'weekday_weekend_correlation': weekday_weekend_corr,
            'daily_total_demand': daily_total_demand
        }

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = DailyPatternAnalyzer()
    
    # 执行完整分析
    results = analyzer.comprehensive_daily_analysis()
    
    print(f"\n每日模式分析完成！主要发现:")
    print(f"- 日间需求模式平均相关系数: {results['avg_correlation']:.4f}")
    print(f"- 工作日与周末模式相关系数: {results['weekday_weekend_correlation']:.4f}")
    print(f"- 分析了 {len(results['daily_total_demand'])} 天的数据")