from load_data import load_data
from docplex.mp.model import Model
import numpy as np
import ast
import time
# 测试不同求解器的速度
methods = {0: "自动", 1: "原始单纯形", 2: "对偶单纯形", 3: "网络单纯形", 4: "内点法", 5: "并发优化", 6: "冲突求解器"}
results = {}

dataset = load_data()
customer_names = dataset['customer_names']
site_names = dataset['site_names']
demand = dataset['demand']
is_link = dataset['is_link']
capacity = dataset['capacity']
time_list = dataset['time_list']

# parameters
big_num = int(1e9) # 大M
I = min(customer_names.size, big_num) # 用户的数量
J = min(site_names.size, big_num) # 站点的数量
T = min(demand.shape[0], 288) # 时间的数量
customer_num = I 
site_num = J 

# y_k，y_k[t,j]表示在时间点t，站点j是否空闲，初始化为零
y_k = np.zeros((T, J), dtype=int)
# free_slots_M8928.txt是一个字典格式的数据，表示不同的时间点t，哪些站点是空闲的
# 字典的键是时间点t，值是一个列表，列表中的元素是站点的索引，这些站点在时间点t是空闲的
free_slots_str = str(np.loadtxt('./test_solution/free_slots_M8928.txt', dtype=str, delimiter=";"))
# 解析字符串为字典
free_slots_dic = ast.literal_eval(free_slots_str)
# 遍历字典，将空闲的站点在对应时间点的y_k设置为1
for t in range(T):
    for site_index in free_slots_dic[t]:
        y_k[t, site_index] = 1

# 构建LP模型
def build_LP():
    mdl = Model(log_output=False)

    # variables
    # 列出所有客户 i、站点 j 和时间点 t 的组合，作为流量变量 X[i, j, t] 的索引
    x_list_index = [(i, j, t) for i in range(I) for j in range(J) for t in range(T)]
    # b定义为每个站点的计费带宽，是一个连续变量，每个站点一个
    b = mdl.continuous_var_list(range(J), name='b')
    # x定义为客户到站点的流量变量，x[i,j,t]表示客户i在时间点t到站点j的流量
    X = mdl.continuous_var_dict(x_list_index, name='X')

    # objective
    # 目标函数是最小化所有站点的计费带宽的总和
    mdl.minimize(mdl.sum(b))

    # constraints
    for t in range(T):
        # 需求约束：每个客户在每个时间点的需求必须满足。对每个客户 i 和时间点 t，将从所有与该客户连接的站点 j 的流量相加，应该等于该客户在该时间点的需求 demand[t, i]。
        for i in range(I):
            mdl.add_constraint(mdl.sum(X[i, j, t] for j in range(J) if is_link[j, i]) == demand[t, i])
        # 容量约束：每个站点 j 在每个时间点 t 的流量不能超过其容量。对于每个站点 j 和时间点 t，如果站点为免费时段（y_k[t, j] == 1），则流量不能超过其容量；如果站点为计费时段（y_k[t, j] == 0），则流量不能超过其计费带宽 b[j]。
        for j in range(J):
            mdl.add_constraint(
                mdl.sum(X[i, j, t] for i in range(I) if is_link[j, i]) <= capacity[j] * y_k[t, j] + (1 - y_k[t, j]) * b[
                    j])
    # mdl.export_as_lp('./experiment_results/LP_model.lp')

    # solve
    mdl.parameters.lpmethod = 4
    print('solve starts...')
    if mdl.solve(log_output=False):
        print('solve time: ', mdl.solve_details.time)
        print('objective: ', mdl.objective_value)
        # mdl.print_solution()
    else:
        print('not solved')

    # ************ 测试各种LP method
    # for method, name in methods.items():
    #     m = mdl.copy()  # 创建模型副本
    #     m.parameters.lpmethod = method
    #     start_time = time.time()
    #     try:
    #         sol = m.solve(log_output=False)
    #         results[name] = time.time() - start_time
    #         status = sol.solve_status if sol else "失败"
    #         print(f"{name}法耗时: {results[name]:.4f}秒，状态: {status}")
    #     except Exception as e:
    #         results[name] = float('inf')
    #         print(f"{name}法求解失败: {str(e)}")


if __name__ == '__main__':
    build_LP()

