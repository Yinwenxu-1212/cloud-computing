"""
基于cplex_MIP.py的松弛LP问题求解器
将原始MIP问题中的二进制变量Y松弛为连续变量[0,1]
使用CPLEX求解器
"""

from load_data import load_data
import numpy as np
import ast
import math
import time
import json
from docplex.mp.model import Model

print("使用CPLEX求解器")

class RelaxedLPSolver:
    def __init__(self):
        self.dataset = load_data()
        self.customer_names = self.dataset['customer_names']
        self.site_names = self.dataset['site_names']
        self.demand = self.dataset['demand']
        self.is_link = self.dataset['is_link']
        self.capacity = self.dataset['capacity']
        self.time_list = self.dataset['time_list']
        
        # 参数设置
        self.big_num = int(1e9)
        self.I = min(self.customer_names.size, self.big_num)  # 客户数
        self.J = min(self.site_names.size, self.big_num)      # 站点数
        self.T = min(self.demand.shape[0], 288)               # 使用第一天数据
        self.L = math.ceil(self.T * 0.05)                     # 每个站点的最大免费时段数
        
        print(f"问题规模: I={self.I}, J={self.J}, T={self.T}, L={self.L}")
        print(f"使用CPLEX求解器")
    
    def solve_with_cplex(self):
        """使用CPLEX求解松弛LP问题"""
        try:
            mdl = Model(log_output=False)
            
            # 变量定义
            x_list_index = [(i, j, t) for i in range(self.I) for j in range(self.J) for t in range(self.T)]
            b = mdl.continuous_var_list(range(self.J), name='b')
            X = mdl.continuous_var_dict(x_list_index, name='X')
            
            # Y变量松弛为连续变量 [0,1]
            y_list_index = [(t, j) for t in range(self.T) for j in range(self.J)]
            Y = mdl.continuous_var_dict(y_list_index, lb=0, ub=1, name='Y')
            
            # 目标函数：最小化计费带宽总和
            mdl.minimize(mdl.sum(b))
            
            # 约束条件
            for t in range(self.T):
                # 需求约束
                for i in range(self.I):
                    mdl.add_constraint(
                        mdl.sum(X[i, j, t] for j in range(self.J) if self.is_link[j, i]) == self.demand[t, i]
                    )
                
                # 容量约束
                for j in range(self.J):
                    # 基础容量约束
                    mdl.add_constraint(
                        mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= self.capacity[j]
                    )
                    # 计费带宽约束
                    mdl.add_constraint(
                        mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= 
                        b[j] + self.capacity[j] * Y[t, j]
                    )
            
            # 免费时段限制约束
            for j in range(self.J):
                mdl.add_constraint(mdl.sum(Y[t, j] for t in range(self.T)) <= self.L)
            
            # 求解
            start_time = time.time()
            solution = mdl.solve()
            solve_time = time.time() - start_time
            
            if solution:
                return {
                    'objective': mdl.objective_value,
                    'solve_time': solve_time,
                    'status': 'optimal',
                    'solver': 'CPLEX',
                    'b_values': [b[j].solution_value for j in range(self.J)],
                    'y_values': {(t, j): Y[t, j].solution_value for t in range(self.T) for j in range(self.J)},
                    'x_values': {(i, j, t): X[i, j, t].solution_value for i in range(self.I) for j in range(self.J) for t in range(self.T)}
                }
            else:
                return {
                    'objective': None,
                    'solve_time': solve_time,
                    'status': 'infeasible',
                    'solver': 'CPLEX'
                }
                
        except Exception as e:
            print(f"CPLEX求解失败: {e}")
            return None
    
    def solve_relaxed_lp(self):
        """求解松弛LP问题"""
        print("开始求解松弛LP问题...")
        
        # 直接使用CPLEX求解
        result = self.solve_with_cplex()
        
        if result is None:
            print("CPLEX求解失败")
            return {
                'objective': None,
                'solve_time': 0,
                'status': 'failed',
                'solver': 'cplex'
            }
        
        print(f"松弛LP求解完成:")
        print(f"  目标值: {result['objective']}")
        print(f"  求解时间: {result['solve_time']:.4f}秒")
        print(f"  状态: {result['status']}")
        print(f"  求解器: {result['solver']}")
        
        return result
    
    def save_results(self, result, filename="relaxed_lp_results.json"):
        """保存结果到文件"""
        if result['objective'] is not None:
            # 转换numpy类型为Python原生类型
            save_data = {
                'objective': float(result['objective']),
                'solve_time': float(result['solve_time']),
                'status': result['status'],
                'solver': result['solver'],
                'problem_size': {
                    'customers': int(self.I),
                    'sites': int(self.J),
                    'time_slots': int(self.T),
                    'max_free_slots': int(self.L)
                }
            }
            
            # 保存完整的Y值
            y_values = {}
            for t in range(self.T):
                for j in range(self.J):
                    if (t, j) in result['y_values']:
                        y_values[f"Y_{t}_{j}"] = float(result['y_values'][(t, j)])
            save_data['y_values'] = y_values
            
            # 保存完整的b值
            if 'b_values' in result:
                save_data['b_values'] = [float(b) for b in result['b_values']]
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存到: {filename}")
        else:
            print("求解失败，无法保存结果")

def main():
    solver = RelaxedLPSolver()
    result = solver.solve_relaxed_lp()
    
    if result['objective'] is not None:
        solver.save_results(result, "experiment_results/relaxed_lp_results.json")
        
        print("\n=== 松弛LP问题求解结果 ===")
        print(f"最优目标值: {result['objective']:.6f}")
        print(f"求解时间: {result['solve_time']:.4f}秒")
        print(f"求解状态: {result['status']}")
        print(f"使用求解器: {result['solver']}")
    else:
        print("求解失败")

if __name__ == "__main__":
    main()