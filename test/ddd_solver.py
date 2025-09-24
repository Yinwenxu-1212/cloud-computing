import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from load_data import load_data
import ast
from docplex.mp.model import Model
import math
@dataclass
class DailySolution:
    """日内求解结果"""
    day: int
    x_solution: np.ndarray  # 流量分配矩阵
    y_solution: Dict[int, List[int]]  # 免费时隙分配
    b_solution: np.ndarray  # 带宽分配
    daily_cost: float
    solve_time: float
    iterations: int
    is_optimal: bool
    solve_info: Optional[Dict] = None  # 详细求解信息，用于95百分位计算

@dataclass
class MonthlySolution:
    """月度求解结果"""
    daily_solutions: List[DailySolution]
    total_cost: float
    percentile_95_bandwidth: float
    solve_time: float
    is_optimal: bool

class TimeGranularityManager:
    """时间粒度管理器"""
    
    def __init__(self, total_slots: int = 288):
        self.total_slots = total_slots  # 一天的5分钟时隙总数
        self.current_granularity = 30  # 当前粒度（分钟）
        self.final_granularity = 5     # 最终粒度（分钟）
        
    def get_coarse_slots(self) -> int:
        """获取当前粒度下的时隙数量"""
        return (24 * 60) // self.current_granularity
    
    def refine_granularity(self):
        """细化时间粒度"""
        if self.current_granularity > self.final_granularity:
            # 逐步细化：30min -> 15min -> 10min -> 5min
            if self.current_granularity == 30:
                self.current_granularity = 15
            elif self.current_granularity == 15:
                self.current_granularity = 10
            elif self.current_granularity == 10:
                self.current_granularity = 5
    
    def map_coarse_to_fine(self, coarse_free_slots_dict: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """将粗粒度免费时隙映射到5分钟精细粒度
        
        Args:
            coarse_free_slots_dict: {coarse_slot: [free_sites]} 粗粒度免费时隙和对应的免费站点
        
        Returns:
            {fine_slot: [free_sites]} 精细粒度免费时隙和对应的免费站点
        """
        fine_free_slots = {}
        slots_per_coarse = self.current_granularity // 5  # 每个粗粒度时隙包含的5分钟时隙数
        
        # 初始化所有精细时隙为空
        for t in range(self.total_slots):
            fine_free_slots[t] = []
        
        # 将粗粒度免费时隙映射到精细粒度
        for coarse_slot, free_sites in coarse_free_slots_dict.items():
            start_fine_slot = coarse_slot * slots_per_coarse
            end_fine_slot = min((coarse_slot + 1) * slots_per_coarse, self.total_slots)
            
            for fine_slot in range(start_fine_slot, end_fine_slot):
                fine_free_slots[fine_slot] = free_sites.copy()
        
        return fine_free_slots

class ADMMSolver:
    """ADMM求解器封装类"""
    
    def __init__(self):
        self.dataset = load_data()
        self.customer_names = self.dataset['customer_names']
        self.site_names = self.dataset['site_names']
        self.demand = self.dataset['demand']
        self.is_link = self.dataset['is_link']
        self.capacity = self.dataset['capacity']
        
        # 系统参数
        big_num = int(1e9)
        self.I = min(self.customer_names.size, big_num)  # 客户数
        self.J = min(self.site_names.size, big_num)     # 站点数
        self.T = min(self.demand.shape[0], 288)         # 时间槽数
        
        # 预计算系统矩阵
        self._build_system_matrices()
    
    def _build_system_matrices(self):
        """构建ADMM求解所需的系统矩阵"""
        # A 矩阵：客户到站点的流量关系
        self.A = torch.zeros((self.J, self.I * self.J), dtype=torch.float32, requires_grad=False)
        for j in range(self.J):
            for i in range(self.I):
                self.A[j, i * self.J + j] = int(self.is_link[j, i])
        
        # B 矩阵：站点到客户的流量关系
        self.B = torch.zeros((self.I, self.I * self.J), dtype=torch.float32, requires_grad=False)
        for i in range(self.I):
            for j in range(self.J):
                self.B[i, i * self.J + j] = int(self.is_link[j, i])
        
        # C 矩阵：站点容量
        self.C = torch.from_numpy(self.capacity[:self.J] / 1000).to(torch.float32)
        self.C.requires_grad = False
        
        # D 矩阵：客户需求
        self.D = torch.from_numpy(self.demand[:self.T, :self.I] / 1000).to(torch.float32)
        self.D.requires_grad = False
        
        # 预计算矩阵运算
        self.AT = self.A.T
        self.BT = self.B.T
        ATA = torch.matmul(self.AT, self.A)
        BTB = torch.matmul(self.BT, self.B)
        diag = torch.eye(self.I * self.J)
        SUM = ATA + BTB + diag
        self.INV = torch.linalg.inv(SUM)
        self.INV.requires_grad = False
    
    def solve_with_fixed_free_slots(self, free_slots: Dict[int, List[int]], 
                                   demand_matrix: torch.Tensor,
                                   rho: float = 0.01, max_iter: int = 500) -> Tuple[float, Dict]:
        """使用固定的免费时隙求解ADMM问题"""
        # 构建 y_k 矩阵
        y_k = torch.zeros((self.T, self.J), dtype=torch.float32, requires_grad=False)
        for t in range(self.T):
            if t in free_slots:
                for site_index in free_slots[t]:
                    if site_index < self.J:
                        y_k[t, site_index] = 1
        
        # 初始化ADMM变量
        x_k1 = torch.zeros((self.I * self.J, self.T), dtype=torch.float32, requires_grad=False)
        l1_k1 = torch.zeros((self.T, self.J), dtype=torch.float32, requires_grad=False)
        l2_k1 = torch.zeros((self.T, self.I), dtype=torch.float32, requires_grad=False)
        l3_k1 = torch.zeros((self.T, self.I * self.J), dtype=torch.float32, requires_grad=False)
        s_k2 = torch.zeros((self.T, self.I * self.J), dtype=torch.float32, requires_grad=False)
        s_k1 = torch.zeros((self.T, self.J), dtype=torch.float32, requires_grad=False)
        b_k1 = torch.zeros(self.J, dtype=torch.float32, requires_grad=False)
        
        # ADMM主循环
        lst_obj = []
        
        with torch.no_grad():
            for iteration in range(max_iter):
                # x 更新
                const_c = ((self.C - b_k1) * y_k + b_k1 - s_k1).to(torch.float32)
                atc = torch.matmul(self.AT, const_c.T)
                btd = torch.matmul(self.BT, demand_matrix.T)
                atl1 = torch.matmul(self.AT, l1_k1.T)
                atl2 = torch.matmul(self.BT, l2_k1.T)
                equ_right = ((atc + btd + s_k2.T) - 1.0 / rho * (atl1 + atl2 + l3_k1.T))
                x_k1[:] = torch.matmul(self.INV, equ_right)
                
                # 计算站点总流量
                A1X = torch.matmul(self.A, x_k1)
                
                # b 更新
                down = rho * torch.sum(torch.square(y_k - 1), dim=0)
                # 避免除零错误
                down = torch.where(down == 0, torch.tensor(1e-8), down)
                sum_x = torch.matmul(self.A, x_k1).T
                up = torch.sum(rho * (y_k - 1) * (self.C * y_k - s_k1 - sum_x - l1_k1/rho), dim=0) - 1
                b_k1[:] = (up / down).clip(0, torch.inf)
                
                # s 更新
                common_term = sum_x - (self.C - b_k1) * y_k - b_k1
                s_k1[:] = (-common_term - l1_k1 / rho).clip(0, torch.inf)
                s_k2[:] = (x_k1.T + l3_k1 / rho).clip(0, torch.inf)
                
                # l 更新
                l1_k1[:] = l1_k1 + rho * (common_term + s_k1)
                sum_x2 = torch.matmul(self.B, x_k1).T
                l2_k1[:] = l2_k1 + rho * (sum_x2 - demand_matrix)
                l3_k1[:] = l3_k1 + rho * (x_k1.T - s_k2)
                
                # 记录目标值
                obj_val = torch.sum(b_k1).cpu().item()
                lst_obj.append(obj_val)
                
                # 收敛检查
                if iteration >= 2 and abs((lst_obj[-1] - lst_obj[-2]) / max(lst_obj[-1], 1e-8)) <= 1e-6:
                    if torch.matmul(self.B, x_k1).ge(demand_matrix.T).all() and \
                       torch.le(torch.floor(A1X.clip(0, torch.inf)), self.C.repeat(self.T, 1).T).all():
                        break
        
        return lst_obj[-1] if lst_obj else 0.0, {
            'x': x_k1.cpu().numpy(),
            'b': b_k1.cpu().numpy(),
            'iterations': len(lst_obj),
            'convergence': lst_obj
        }

class DailyOptimalSolver:
    """算法2：日内最优求解器 (DDD核心框架)"""
    
    def __init__(self, tolerance: float = 0.0001):
        self.tolerance = tolerance
        self.admm_solver = ADMMSolver()
        
    def solve_daily(self, day_demand: np.ndarray, day: int) -> DailySolution:
        """求解单日最优带宽分配"""
        start_time = time.time()
        
        # 初始化时间粒度管理器
        granularity_manager = TimeGranularityManager()
        
        # 初始化上下界
        UB = float('inf')
        LB = float('-inf')
        best_solution = None
        iteration = 0
        
        print(f"\n开始求解第{day}天的日内优化问题")
        print(f"初始时间粒度: {granularity_manager.current_granularity}分钟")
        
        # 迭代细化循环
        while True:
            iteration += 1
            print(f"\n--- 日内迭代 {iteration} ---")
            print(f"当前时间粒度: {granularity_manager.current_granularity}分钟")
            
            # 步骤1: 下界计算 - 求解粗粒度MIP问题
            current_LB, coarse_solution = self._solve_coarse_mip(granularity_manager, day_demand)
            LB = max(LB, current_LB)
            print(f"粗粒度MIP下界: {current_LB:.4f}")
            
            # 步骤2: 上界计算 - 修复解并评估真实成本
            current_UB, fine_solution = self._repair_and_evaluate(coarse_solution, granularity_manager, day_demand)
            print(f"修复后上界: {current_UB:.4f}")
            
            if current_UB < UB:
                UB = current_UB
                best_solution = fine_solution
            
            # 步骤3: 最优性检验
            if UB == 0:
                gap = 0
            else:
                gap = (UB - LB) / UB
            print(f"最优性间隙: {gap:.6f}")
            
            if gap <= self.tolerance:
                print("达到最优性条件！")
                break
            
            # 步骤4: 模型细化
            if granularity_manager.current_granularity <= granularity_manager.final_granularity:
                print("已达到最细粒度，停止迭代")
                break
                
            granularity_manager.refine_granularity()
            print(f"细化到新粒度: {granularity_manager.current_granularity}分钟")
        
        solve_time = time.time() - start_time
        
        return DailySolution(
            day=day,
            x_solution=best_solution['x'] if best_solution else np.array([]),
            y_solution=best_solution['free_slots'] if best_solution else {},
            b_solution=best_solution['b'] if best_solution else np.array([]),
            daily_cost=UB,
            solve_time=solve_time,
            iterations=iteration,
            is_optimal=gap <= self.tolerance,
            solve_info=best_solution['solve_info'] if best_solution else None
        )
    
    def _solve_coarse_mip(self, granularity_manager: TimeGranularityManager, day_demand: np.ndarray) -> Tuple[float, Dict]:
        """求解粗粒度MIP问题
        
        使用CPLEX构建真正的MIP模型来求解粗粒度优化问题
        """
        coarse_slots = granularity_manager.get_coarse_slots()
        num_sites = self.admm_solver.J
        num_customers = self.admm_solver.I
        
        # 聚合需求到粗粒度时隙
        slots_per_coarse = granularity_manager.current_granularity // granularity_manager.final_granularity
        coarse_demand = np.zeros((coarse_slots, num_customers))
        
        for t_coarse in range(coarse_slots):
            start_idx = t_coarse * slots_per_coarse
            end_idx = min(start_idx + slots_per_coarse, day_demand.shape[0])
            coarse_demand[t_coarse, :] = np.mean(day_demand[start_idx:end_idx, :], axis=0)
        
        # 构建CPLEX MIP模型
        mdl = Model(log_output=False)
        
        # 决策变量
        # b: 每个站点的计费带宽
        b = mdl.continuous_var_list(range(num_sites), name='b', lb=0)
        
        # X: 客户i在站点j时间点t的流量分配
        X = mdl.continuous_var_dict(
            [(i, j, t) for i in range(num_customers) for j in range(num_sites) for t in range(coarse_slots)],
            name='X', lb=0
        )
        
        # Y: 站点j在时间点t是否为免费时段（1表示免费，0表示计费）
        Y = mdl.binary_var_dict(
            [(t, j) for t in range(coarse_slots) for j in range(num_sites)],
            name='Y'
        )
        
        # 目标函数：最小化总计费带宽
        mdl.minimize(mdl.sum(b))
        
        # 约束条件
        # 1. 需求满足约束
        for t in range(coarse_slots):
            for i in range(num_customers):
                connected_sites = [j for j in range(num_sites) if self.admm_solver.is_link[j, i]]
                if connected_sites:
                    mdl.add_constraint(
                        mdl.sum(X[i, j, t] for j in connected_sites) == coarse_demand[t, i]
                    )
        
        # 2. 容量约束
        slots_per_coarse = granularity_manager.current_granularity // 5  # 每个粗粒度时隙包含的5分钟时隙数
        
        for j in range(num_sites):
            connected_customers = [i for i in range(num_customers) if self.admm_solver.is_link[j, i]]
            if connected_customers:
                # 对每个粗粒度时隙，约束该时隙内的平均流量（平均到5分钟）
                for t in range(coarse_slots):
                    slot_flow = mdl.sum(X[i, j, t] for i in connected_customers)
                    # 粗粒度时隙内的平均流量（平均到5分钟时隙）
                    avg_flow_per_5min = slot_flow / slots_per_coarse
                    
                    # 平均流量不超过站点容量
                    mdl.add_constraint(avg_flow_per_5min <= self.admm_solver.capacity[j])
                    
                    # 如果该时隙免费，平均流量可以达到容量；否则受计费带宽限制
                    mdl.add_constraint(
                        avg_flow_per_5min <= b[j] + self.admm_solver.capacity[j] * Y[t, j]
                    )
        
        # 3. 免费时段限制
        L = math.ceil(coarse_slots * 0.05)  # 每个站点最多5%的时段可以免费
        for j in range(num_sites):
            mdl.add_constraint(mdl.sum(Y[t, j] for t in range(coarse_slots)) <= L)
        
        # 求解模型
        solution = mdl.solve()
        
        if solution:
            obj_val = mdl.objective_value
            
            # 提取免费时段配置
            free_slots_dict = {}
            for t in range(coarse_slots):
                free_sites = [j for j in range(num_sites) if Y[t, j].solution_value > 0.5]
                if free_sites:
                    free_slots_dict[t] = free_sites
            
            return obj_val, {'coarse_free_slots': free_slots_dict}
        else:
            # 如果求解失败，返回一个保守的解
            print("CPLEX求解失败，使用保守估计")
            return float('inf'), {'coarse_free_slots': {}}
    

    
    def _repair_and_evaluate(self, coarse_solution: Dict, granularity_manager: TimeGranularityManager, day_demand: np.ndarray) -> Tuple[float, Dict]:
        """修复解并评估真实成本"""
        coarse_free_slots_dict = coarse_solution['coarse_free_slots']
        fine_free_slots = granularity_manager.map_coarse_to_fine(coarse_free_slots_dict)
        
        # 将需求数据转换为torch.Tensor格式
        demand_tensor = torch.from_numpy(day_demand / 1000).float()
        
        # 在5分钟精细粒度上求解TR问题
        obj_val, solve_info = self.admm_solver.solve_with_fixed_free_slots(fine_free_slots, demand_tensor)
        
        return obj_val, {
            'x': solve_info['x'],
            'b': solve_info['b'],
            'free_slots': fine_free_slots,
            'solve_info': solve_info
        }

class MonthlyPlanGenerator:
    """算法1：主算法 - 月度计划生成器"""
    
    def __init__(self, tolerance: float = 0.0001):
        self.daily_solver = DailyOptimalSolver(tolerance)
        self.dataset = load_data()
        
    def generate_monthly_plan(self, days_in_month: int = 31) -> MonthlySolution:
        """生成月度最优计划"""
        start_time = time.time()
        daily_solutions = []
        
        print(f"开始生成{days_in_month}天的月度计划")
        
        # 按天循环求解
        for day in range(1, days_in_month + 1):
            print(f"\n{'='*50}")
            print(f"处理第{day}天 ({day}/{days_in_month})")
            print(f"{'='*50}")
            
            # 提取当天的流量需求数据
            day_demand = self._extract_daily_demand(day)
            
            # 调用日内最优求解器
            daily_solution = self.daily_solver.solve_daily(day_demand, day)
            daily_solutions.append(daily_solution)
            
            print(f"第{day}天求解完成:")
            print(f"  - 日内成本: {daily_solution.daily_cost:.4f}")
            print(f"  - 求解时间: {daily_solution.solve_time:.2f}秒")
            print(f"  - 是否最优: {daily_solution.is_optimal}")
        
        # 汇总与评估
        total_cost, percentile_95 = self._calculate_monthly_metrics(daily_solutions)
        solve_time = time.time() - start_time
        
        is_optimal = all(sol.is_optimal for sol in daily_solutions)
        
        print(f"\n{'='*50}")
        print(f"月度计划生成完成")
        print(f"{'='*50}")
        print(f"月度总成本: {total_cost:.4f}")
        print(f"95百分位带宽: {percentile_95:.4f}")
        print(f"总求解时间: {solve_time:.2f}秒")
        print(f"所有日内问题是否最优: {is_optimal}")
        
        return MonthlySolution(
            daily_solutions=daily_solutions,
            total_cost=total_cost,
            percentile_95_bandwidth=percentile_95,
            solve_time=solve_time,
            is_optimal=is_optimal
        )
    
    def _extract_daily_demand(self, day: int) -> np.ndarray:
        """提取指定天的需求数据"""
        # 每天288个时隙（5分钟间隔），从第day天开始提取
        start_idx = (day - 1) * 288
        end_idx = day * 288
        
        # 确保不超出数据范围
        total_slots = self.dataset['demand'].shape[0]
        if start_idx >= total_slots:
            # 如果超出范围，使用最后一天的数据
            start_idx = total_slots - 288
            end_idx = total_slots
        elif end_idx > total_slots:
            end_idx = total_slots
            
        return self.dataset['demand'][start_idx:end_idx, :]
    
    def _calculate_monthly_metrics(self, daily_solutions: List[DailySolution]) -> Tuple[float, float]:
        """计算月度指标"""
        # 月度总成本计算：这里应该是95百分位带宽成本，而不是各日成本之和
        # 根据BACE-95问题定义，需要计算所有时隙中95百分位的带宽成本
        
        # 收集所有时隙的实际带宽使用量
        all_bandwidth_usage = []
        
        for sol in daily_solutions:
              if sol.solve_info and 'x' in sol.solve_info:
                  # 从TR问题的解中提取每个时隙的实际带宽使用量
                  x_solution = sol.solve_info['x']  # (I*J, T)
                  # 计算每个站点在每个时隙的流量总和
                  A = self.daily_solver.admm_solver.A  # (J, I*J)
                  site_flows = np.matmul(A.cpu().numpy(), x_solution)  # (J, T)
                  
                  # 计算每个时隙的最大站点带宽使用量（简化处理）
                  max_bandwidth_per_slot = np.max(site_flows, axis=0)  # (T,)
                  all_bandwidth_usage.extend(max_bandwidth_per_slot[max_bandwidth_per_slot > 0])
              elif len(sol.b_solution) > 0:
                  # 如果没有详细信息，使用b_solution作为近似
                  all_bandwidth_usage.extend(sol.b_solution)
        
        if all_bandwidth_usage:
            # 计算95百分位带宽
            percentile_95 = np.percentile(all_bandwidth_usage, 95)
            # BACE-95的总成本就是95百分位带宽成本
            total_cost = percentile_95
        else:
            percentile_95 = 0.0
            total_cost = 0.0
        
        return total_cost, percentile_95

def test_monthly_plan_generator():
    """测试月度计划生成器"""
    print("开始测试月度计划生成器...")
    
    # 为了测试，只处理3天
    generator = MonthlyPlanGenerator(tolerance=0.01)
    monthly_solution = generator.generate_monthly_plan(days_in_month=3)
    
    print(f"\n=== 最终结果汇总 ===")
    print(f"处理天数: {len(monthly_solution.daily_solutions)}")
    print(f"月度总成本: {monthly_solution.total_cost:.4f}")
    print(f"95百分位带宽: {monthly_solution.percentile_95_bandwidth:.4f}")
    print(f"总求解时间: {monthly_solution.solve_time:.2f}秒")
    print(f"整体是否最优: {monthly_solution.is_optimal}")
    
    # 显示每日详情
    print(f"\n=== 每日求解详情 ===")
    for sol in monthly_solution.daily_solutions:
        print(f"第{sol.day}天: 成本={sol.daily_cost:.4f}, 时间={sol.solve_time:.2f}s, 最优={sol.is_optimal}")

if __name__ == "__main__":
    test_monthly_plan_generator()