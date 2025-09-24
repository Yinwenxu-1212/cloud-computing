"""
Gap分析脚本：对比ADMM算法结果与LP松弛解
分析ADMM算法的优化质量和收敛性能
"""

from load_data import load_data
from docplex.mp.model import Model
import numpy as np
import ast
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 设置中文字体和警告过滤
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GapAnalyzer:
    def __init__(self):
        self.dataset = load_data()
        self.customer_names = self.dataset['customer_names']
        self.site_names = self.dataset['site_names']
        self.demand = self.dataset['demand']
        self.is_link = self.dataset['is_link']
        self.capacity = self.dataset['capacity']
        self.time_list = self.dataset['time_list']
        
        # 参数设置
        big_num = int(1e9)
        self.I = min(self.customer_names.size, big_num)  # 客户数
        self.J = min(self.site_names.size, big_num)      # 站点数
        self.T = min(self.demand.shape[0], 288)          # 使用第一天数据
        
        # 加载免费时段数据
        self.y_k = self._load_free_slots()
        
        # 存储结果
        self.results = {}
        
    def _load_free_slots(self):
        """加载免费时段数据"""
        y_k = np.zeros((self.T, self.J), dtype=int)
        free_slots_str = str(np.loadtxt('./test_solution/free_slots_M8928.txt', dtype=str, delimiter=";"))
        free_slots_dic = ast.literal_eval(free_slots_str)
        
        for t in range(self.T):
            for site_index in free_slots_dic[t]:
                y_k[t, site_index] = 1
                
        return y_k
    
    def solve_relaxed_lp(self):
        """求解松弛LP问题（将yjt松弛为连续变量）"""
        print("=== 求解松弛LP问题 ===")
        
        mdl = Model(log_output=False)
        
        # 设置CPLEX参数
        try:
            mdl.context.cplex_parameters.lpmethod = 4  # 使用内点法
            mdl.context.cplex_parameters.threads = 1
        except:
            pass  # 如果设置失败，使用默认参数
        
        # 变量定义
        x_list_index = [(i, j, t) for i in range(self.I) for j in range(self.J) for t in range(self.T)]
        y_list_index = [(t, j) for t in range(self.T) for j in range(self.J)]
        
        # 连续变量
        b = mdl.continuous_var_list(range(self.J), name='b', lb=0)  # 计费带宽
        X = mdl.continuous_var_dict(x_list_index, name='X', lb=0)   # 流量变量
        Y = mdl.continuous_var_dict(y_list_index, name='Y', lb=0, ub=1)  # 松弛的二进制变量
        
        # 目标函数：最小化计费带宽总和
        mdl.minimize(mdl.sum(b))
        
        # 约束条件
        for t in range(self.T):
            # 需求约束：每个客户的需求必须满足
            for i in range(self.I):
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for j in range(self.J) if self.is_link[j, i]) == self.demand[t, i]
                )
            
            # 容量约束：使用与relaxed_lp_solver.py相同的线性化约束
            for j in range(self.J):
                # 基本容量约束
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= self.capacity[j]
                )
                
                # 计费带宽约束（使用大M方法）
                M = self.capacity[j]  # 使用容量作为大M
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= 
                    b[j] + M * Y[t, j]
                )
        
        # 免费时段限制：每个站点最多5%的时间可以免费
        L = int(np.ceil(self.T * 0.05))
        for j in range(self.J):
            mdl.add_constraint(mdl.sum(Y[t, j] for t in range(self.T)) <= L)
        
        # 求解
        start_time = time.time()
        solution = mdl.solve(log_output=False)
        solve_time = time.time() - start_time
        
        if solution:
            lp_objective = mdl.objective_value
            print(f"松弛LP求解成功")
            print(f"求解时间: {solve_time:.4f}秒")
            print(f"目标值: {lp_objective:.6f}")
            
            # 提取解
            b_values = [b[j].solution_value for j in range(self.J)]
            y_values = [[Y[t, j].solution_value for j in range(self.J)] for t in range(self.T)]
            
            self.results['lp_relaxed'] = {
                'objective': lp_objective,
                'solve_time': solve_time,
                'b_values': b_values,
                'y_values': y_values,
                'status': 'optimal'
            }
            
            return lp_objective, b_values, y_values
        else:
            print("松弛LP求解失败")
            self.results['lp_relaxed'] = {
                'objective': None,
                'solve_time': solve_time,
                'status': 'failed'
            }
            return None, None, None
    
    def solve_fixed_y_lp(self):
        """求解固定Y的LP问题（使用给定的免费时段）"""
        print("=== 求解固定Y的LP问题 ===")
        
        mdl = Model(log_output=False)
        
        # 设置CPLEX参数
        try:
            mdl.context.cplex_parameters.lpmethod = 4  # 使用内点法
            mdl.context.cplex_parameters.threads = 1
        except:
            pass  # 如果设置失败，使用默认参数
        
        # 变量定义
        x_list_index = [(i, j, t) for i in range(self.I) for j in range(self.J) for t in range(self.T)]
        
        # 连续变量
        b = mdl.continuous_var_list(range(self.J), name='b', lb=0)
        X = mdl.continuous_var_dict(x_list_index, name='X', lb=0)
        
        # 目标函数
        mdl.minimize(mdl.sum(b))
        
        # 约束条件
        for t in range(self.T):
            # 需求约束
            for i in range(self.I):
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for j in range(self.J) if self.is_link[j, i]) == self.demand[t, i]
                )
            
            # 容量约束（使用固定的Y值）
            for j in range(self.J):
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= 
                    self.capacity[j] * self.y_k[t, j] + b[j] * (1 - self.y_k[t, j])
                )
        
        # 求解
        start_time = time.time()
        solution = mdl.solve(log_output=False)
        solve_time = time.time() - start_time
        
        if solution:
            lp_fixed_objective = mdl.objective_value
            print(f"固定Y的LP求解成功")
            print(f"求解时间: {solve_time:.4f}秒")
            print(f"目标值: {lp_fixed_objective:.6f}")
            
            b_values = [b[j].solution_value for j in range(self.J)]
            
            self.results['lp_fixed_y'] = {
                'objective': lp_fixed_objective,
                'solve_time': solve_time,
                'b_values': b_values,
                'status': 'optimal'
            }
            
            return lp_fixed_objective, b_values
        else:
            print("固定Y的LP求解失败")
            self.results['lp_fixed_y'] = {
                'objective': None,
                'solve_time': solve_time,
                'status': 'failed'
            }
            return None, None
    
    def load_admm_results(self, rho=0.01):
        """加载ADMM算法结果"""
        print("=== 加载ADMM算法结果 ===")
        
        try:
            filename = f'experiment_results/RHO={rho}.txt'
            admm_objectives = np.loadtxt(filename)
            
            # 取最后一个值作为ADMM的最终结果
            admm_final_objective = admm_objectives[-1]
            
            print(f"ADMM算法结果加载成功")
            print(f"迭代次数: {len(admm_objectives)}")
            print(f"最终目标值: {admm_final_objective:.6f}")
            print(f"收敛情况: {'收敛' if len(admm_objectives) < 500 else '未完全收敛'}")
            
            self.results['admm'] = {
                'objective': admm_final_objective,
                'iterations': len(admm_objectives),
                'objectives_history': admm_objectives.tolist(),
                'rho': rho,
                'converged': len(admm_objectives) < 500
            }
            
            return admm_final_objective, admm_objectives
        except Exception as e:
            print(f"ADMM结果加载失败: {e}")
            return None, None
    
    def calculate_gaps(self):
        """计算各种gap"""
        print("=== 计算Gap分析 ===")
        
        if 'admm' not in self.results or 'lp_fixed_y' not in self.results:
            print("缺少必要的求解结果，无法计算gap")
            return
        
        admm_obj = self.results['admm']['objective']
        lp_fixed_obj = self.results['lp_fixed_y']['objective']
        
        gaps = {}
        
        # ADMM与固定Y的LP的gap
        if lp_fixed_obj is not None and lp_fixed_obj > 0:
            gap_fixed = (admm_obj - lp_fixed_obj) / lp_fixed_obj * 100
            gaps['admm_vs_lp_fixed'] = gap_fixed
            print(f"ADMM vs 固定Y的LP gap: {gap_fixed:.4f}%")
        
        # 如果有松弛LP结果
        if 'lp_relaxed' in self.results and self.results['lp_relaxed']['objective'] is not None:
            lp_relaxed_obj = self.results['lp_relaxed']['objective']
            if lp_relaxed_obj > 0:
                gap_relaxed = (admm_obj - lp_relaxed_obj) / lp_relaxed_obj * 100
                gaps['admm_vs_lp_relaxed'] = gap_relaxed
                print(f"ADMM vs 松弛LP gap: {gap_relaxed:.4f}%")
                
                # LP松弛与固定Y LP的gap
                if lp_fixed_obj is not None and lp_fixed_obj > 0:
                    gap_lp_bound = (lp_fixed_obj - lp_relaxed_obj) / lp_relaxed_obj * 100
                    gaps['lp_fixed_vs_lp_relaxed'] = gap_lp_bound
                    print(f"固定Y LP vs 松弛LP gap: {gap_lp_bound:.4f}%")
        
        self.results['gaps'] = gaps
        return gaps
    
    def plot_convergence_analysis(self):
        """绘制收敛性分析图"""
        print("=== 绘制收敛性分析 ===")
        
        if 'admm' not in self.results:
            print("缺少ADMM结果，无法绘制收敛图")
            return
        
        admm_history = self.results['admm']['objectives_history']
        
        plt.figure(figsize=(15, 10))
        
        # 子图1：ADMM收敛曲线
        plt.subplot(2, 2, 1)
        plt.plot(admm_history, 'b-', linewidth=2, label='ADMM目标值')
        
        # 添加基准线
        if 'lp_fixed_y' in self.results and self.results['lp_fixed_y']['objective'] is not None:
            plt.axhline(y=self.results['lp_fixed_y']['objective'], color='r', 
                       linestyle='--', linewidth=2, label='固定Y LP最优解')
        
        if 'lp_relaxed' in self.results and self.results['lp_relaxed']['objective'] is not None:
            plt.axhline(y=self.results['lp_relaxed']['objective'], color='g', 
                       linestyle='--', linewidth=2, label='松弛LP下界')
        
        plt.xlabel('迭代次数')
        plt.ylabel('目标值')
        plt.title('ADMM算法收敛曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：收敛速度（相对变化）
        plt.subplot(2, 2, 2)
        if len(admm_history) > 1:
            relative_changes = [abs((admm_history[i] - admm_history[i-1]) / admm_history[i-1]) 
                              for i in range(1, len(admm_history))]
            plt.semilogy(relative_changes, 'r-', linewidth=2)
            plt.axhline(y=1e-6, color='k', linestyle='--', alpha=0.5, label='收敛阈值')
            plt.xlabel('迭代次数')
            plt.ylabel('相对变化 (log scale)')
            plt.title('ADMM收敛速度')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 子图3：Gap对比
        plt.subplot(2, 2, 3)
        if 'gaps' in self.results:
            gaps = self.results['gaps']
            gap_names = []
            gap_values = []
            
            if 'admm_vs_lp_fixed' in gaps:
                gap_names.append('ADMM vs\n固定Y LP')
                gap_values.append(gaps['admm_vs_lp_fixed'])
            
            if 'admm_vs_lp_relaxed' in gaps:
                gap_names.append('ADMM vs\n松弛LP')
                gap_values.append(gaps['admm_vs_lp_relaxed'])
            
            if 'lp_fixed_vs_lp_relaxed' in gaps:
                gap_names.append('固定Y LP vs\n松弛LP')
                gap_values.append(gaps['lp_fixed_vs_lp_relaxed'])
            
            if gap_names:
                colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(gap_names)]
                bars = plt.bar(gap_names, gap_values, color=colors, alpha=0.7)
                plt.ylabel('Gap (%)')
                plt.title('优化Gap对比')
                plt.grid(True, alpha=0.3)
                
                # 在柱状图上添加数值标签
                for bar, value in zip(bars, gap_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}%', ha='center', va='bottom')
        
        # 子图4：目标值对比
        plt.subplot(2, 2, 4)
        methods = []
        objectives = []
        
        if 'lp_relaxed' in self.results and self.results['lp_relaxed']['objective'] is not None:
            methods.append('松弛LP\n(下界)')
            objectives.append(self.results['lp_relaxed']['objective'])
        
        if 'lp_fixed_y' in self.results and self.results['lp_fixed_y']['objective'] is not None:
            methods.append('固定Y LP')
            objectives.append(self.results['lp_fixed_y']['objective'])
        
        if 'admm' in self.results:
            methods.append('ADMM')
            objectives.append(self.results['admm']['objective'])
        
        if methods:
            colors = ['lightgreen', 'skyblue', 'lightcoral'][:len(methods)]
            bars = plt.bar(methods, objectives, color=colors, alpha=0.7)
            plt.ylabel('目标值')
            plt.title('不同方法目标值对比')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, objectives):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('experiment_results/gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """生成详细的分析报告"""
        print("=== 生成分析报告 ===")
        
        report = f"""
# ADMM算法与最优解Gap分析报告

## 1. 分析概述
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 问题规模: {self.I}个客户, {self.J}个站点, {self.T}个时间点
- 免费时段比例: {np.sum(self.y_k) / (self.T * self.J) * 100:.2f}%

## 2. 求解结果对比

### 2.1 ADMM算法结果
"""
        
        if 'admm' in self.results:
            admm = self.results['admm']
            report += f"""
- 目标值: {admm['objective']:.6f}
- 迭代次数: {admm['iterations']}
- 收敛状态: {'收敛' if admm['converged'] else '未完全收敛'}
- 罚参数ρ: {admm['rho']}
"""
        
        if 'lp_fixed_y' in self.results:
            lp_fixed = self.results['lp_fixed_y']
            report += f"""
### 2.2 固定Y的LP问题结果
- 目标值: {lp_fixed['objective']:.6f}
- 求解时间: {lp_fixed['solve_time']:.4f}秒
- 求解状态: {lp_fixed['status']}
"""
        
        if 'lp_relaxed' in self.results:
            lp_relaxed = self.results['lp_relaxed']
            report += f"""
### 2.3 松弛LP问题结果
- 目标值: {lp_relaxed['objective']:.6f}
- 求解时间: {lp_relaxed['solve_time']:.4f}秒
- 求解状态: {lp_relaxed['status']}
"""
        
        if 'gaps' in self.results:
            gaps = self.results['gaps']
            report += f"""
## 3. Gap分析

### 3.1 优化质量评估
"""
            
            if 'admm_vs_lp_fixed' in gaps and gaps['admm_vs_lp_fixed'] is not None:
                gap = gaps['admm_vs_lp_fixed']
                quality = "优秀" if gap < 1 else "良好" if gap < 5 else "一般" if gap < 10 else "较差"
                report += f"""
- **ADMM vs 固定Y LP**: {gap:.4f}%
  - 评估: {quality}
  - 说明: ADMM算法相对于给定免费时段下的最优解的gap
"""
            
            if 'admm_vs_lp_relaxed' in gaps and gaps['admm_vs_lp_relaxed'] is not None:
                gap = gaps['admm_vs_lp_relaxed']
                report += f"""
- **ADMM vs 松弛LP**: {gap:.4f}%
  - 说明: ADMM算法相对于理论下界的gap
"""
            
            if 'lp_fixed_vs_lp_relaxed' in gaps and gaps['lp_fixed_vs_lp_relaxed'] is not None:
                gap = gaps['lp_fixed_vs_lp_relaxed']
                report += f"""
- **固定Y LP vs 松弛LP**: {gap:.4f}%
  - 说明: 固定免费时段策略的优化空间
"""
        
        report += f"""
## 4. 结论与建议

### 4.1 主要发现
"""
        
        if 'gaps' in self.results and 'admm_vs_lp_fixed' in self.results['gaps']:
            gap = self.results['gaps']['admm_vs_lp_fixed']
            if gap < 1:
                report += "1. **ADMM算法表现优秀**: gap小于1%，接近最优解\n"
            elif gap < 5:
                report += "1. **ADMM算法表现良好**: gap在可接受范围内\n"
            else:
                report += "1. **ADMM算法有改进空间**: gap较大，可能需要调整参数\n"
        
        if 'admm' in self.results:
            if self.results['admm']['converged']:
                report += "2. **收敛性良好**: ADMM算法在合理迭代次数内收敛\n"
            else:
                report += "2. **收敛性需要改进**: 可能需要调整罚参数或增加迭代次数\n"
        
        report += f"""
### 4.2 优化建议
1. **参数调优**: 尝试不同的罚参数ρ值
2. **收敛准则**: 调整收敛阈值以平衡精度和效率
3. **初始化策略**: 使用更好的初始解
4. **算法改进**: 考虑自适应ADMM或其他变种

## 5. 生成文件
- gap_analysis.png: Gap分析可视化图表
- gap_analysis_report.txt: 本分析报告
- gap_analysis_results.json: 详细数值结果
"""
        
        # 保存报告
        with open('experiment_results/gap_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON结果
        with open('experiment_results/gap_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("分析报告已保存到 experiment_results/gap_analysis_report.txt")
        print("详细结果已保存到 experiment_results/gap_analysis_results.json")
        print(report)
    
    def run_complete_analysis(self, rho=0.01):
        """运行完整的gap分析"""
        print("=== 开始完整Gap分析 ===")
        
        # 1. 加载ADMM结果
        self.load_admm_results(rho)
        
        # 2. 求解固定Y的LP问题
        self.solve_fixed_y_lp()
        
        # 3. 求解松弛LP问题
        self.solve_relaxed_lp()
        
        # 4. 计算gap
        self.calculate_gaps()
        
        # 5. 绘制分析图
        self.plot_convergence_analysis()
        
        # 6. 生成报告
        self.generate_report()
        
        return self.results

if __name__ == "__main__":
    analyzer = GapAnalyzer()
    results = analyzer.run_complete_analysis(rho=0.01)
    
    print("\n=== Gap分析完成 ===")
    if 'gaps' in results:
        for gap_name, gap_value in results['gaps'].items():
            print(f"{gap_name}: {gap_value:.4f}%")