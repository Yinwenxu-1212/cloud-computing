import time
import math
import numpy as np
from docplex.mp.model import Model
from load_data import load_data

class LPVerifier:
    """
    一个用于验证松弛LP问题性质的类。
    性质：当最优目标值为0时，每个数据中心的平均流量应小于等于其容量的5%。
    """
    def __init__(self):
        """初始化，加载数据并设置模型参数。"""
        print("1. 初始化并加载数据...")
        self.dataset = load_data()
        self.capacity = self.dataset['capacity']
        self.demand = self.dataset['demand']
        self.is_link = self.dataset['is_link']
        
        # 参数设置
        self.I = self.dataset['customer_names'].size  # 客户数
        self.J = self.dataset['site_names'].size      # 站点数
        self.T = min(self.demand.shape[0], 288)       # 使用第一天数据
        self.L = math.ceil(self.T * 0.05)             # 每个站点的最大免费时段数
        
        print(f"问题规模: I={self.I}, J={self.J}, T={self.T}, L={self.L}")
        print("-" * 40)

    def solve_relaxed_lp(self):
        """
        构建并求解松弛LP模型。
        使用会导致目标值为0的“弱”松弛约束，以验证理论。
        """
        print("2. 开始求解松弛LP问题...")
        mdl = Model(name='Relaxed_LP_Verification', log_output=False)
        
        # 变量定义
        b = mdl.continuous_var_list(range(self.J), name='b', lb=0)
        X = mdl.continuous_var_dict(
            [(i, j, t) for i in range(self.I) for j in range(self.J) for t in range(self.T)], 
            name='X', lb=0
        )
        Y = mdl.continuous_var_dict(
            [(t, j) for t in range(self.T) for j in range(self.J)], 
            name='Y', lb=0, ub=1
        )
        
        # 目标函数
        mdl.minimize(mdl.sum(b))
        
        # 约束条件
        for t in range(self.T):
            for i in range(self.I):
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for j in range(self.J) if self.is_link[j, i]) == self.demand[t, i]
                )
            for j in range(self.J):
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= self.capacity[j]
                )
                # 使用“弱”的计费带宽约束
                mdl.add_constraint(
                    mdl.sum(X[i, j, t] for i in range(self.I) if self.is_link[j, i]) <= 
                    b[j] + self.capacity[j] * Y[t, j]
                )
        
        for j in range(self.J):
            mdl.add_constraint(mdl.sum(Y[t, j] for t in range(self.T)) == self.L)
            
        # 求解
        start_time = time.time()
        solution = mdl.solve()
        solve_time = time.time() - start_time
        
        print(f"求解完成，耗时: {solve_time:.4f}秒")
        
        if solution:
            print(f"求解成功，目标值: {solution.get_objective_value():.6f}")
            return solution, X
        else:
            print("求解失败！")
            return None, None
            
    def verify_property(self):
        """
        主函数，执行求解和验证过程。
        """
        solution, X = self.solve_relaxed_lp()
        
        if not solution:
            print("无法进行验证，因为模型求解失败。")
            return
            
        objective_value = solution.get_objective_value()
        if not math.isclose(objective_value, 0, abs_tol=1e-5):
            print(f"警告：目标值 {objective_value} 不为0，验证可能无意义。")

        print("\n3. 开始验证理论性质...")
        print("性质：每个数据中心的优化后平均流量应 ≤ 容量 × 5%")
        print("-" * 70)
        
        header = f"{'数据中心':>10} {'优化后平均流量':>18} {'5%容量阈值':>15} {'比值':>10} {'状态':>8}"
        print(header)
        print("-" * 70)
        
        violation_count = 0
        
        x_values = solution.get_value_dict(X)

        for j in range(self.J):
            total_traffic_volume = 0
            for t in range(self.T):
                traffic_at_t = sum(x_values.get((i, j, t), 0) for i in range(self.I) if self.is_link[j, i])
                total_traffic_volume += traffic_at_t
            
            avg_traffic = total_traffic_volume / self.T
            
            capacity_threshold = self.capacity[j] * (self.L / self.T)
            
            ratio = avg_traffic / capacity_threshold if capacity_threshold > 0 else 0
            status = "✓" if avg_traffic <= capacity_threshold + 1e-6 else "✗ (违反)"
            
            if "✗" in status:
                violation_count += 1
            
            print(f"{j:>10} {avg_traffic:>18.2f} {capacity_threshold:>15.2f} {ratio:>10.3f} {status:>8}")

        print("-" * 70)
        print(f"验证完成。")
        if violation_count == 0:
            print(f"✓ 结果正确：所有 {self.J}/{self.J} 个数据中心都满足该性质。")
            print("这证明了求解器通过重新分配流量，使得所有b_j=0成为可能。")
        else:
            print(f"✗ 结果不符：有 {violation_count}/{self.J} 个数据中心违反了该性质。")
            print("这表明存在理论或实现上的偏差，请检查模型约束。")

if __name__ == "__main__":
    verifier = LPVerifier()
    verifier.verify_property()