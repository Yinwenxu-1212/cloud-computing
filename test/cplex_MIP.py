from load_data import load_data
from docplex.mp.model import Model
import numpy as np
import ast
import math

dataset = load_data()
customer_names = dataset['customer_names']
site_names = dataset['site_names']
demand = dataset['demand']
is_link = dataset['is_link']
capacity = dataset['capacity']
time_list = dataset['time_list']

# parameters
big_num = int(1e9) # 大M
# 客户数
I = min(customer_names.size, big_num)
# 站点数
J = min(site_names.size, big_num)
# 时间点数
T = min(demand.shape[0], 48)
# 客户数
customer_num = I
# 站点数
site_num = J
# 每个站点的最大免费时段数
L = math.ceil(T * 0.05)



def build_MIP():
    mdl = Model(log_output=False)

    # Benders parameters
    # mdl.parameters.benders.strategy = 3  # force Benders
    # mdl.parameters.benders.strategy.set(mdl.parameters.benders.strategy.values.full)

    # variables
    x_list_index = [(i, j, t) for i in range(I) for j in range(J) for t in range(T)]
    # b：定义每个站点的计费带宽变量，b[j] 为连续变量
    b = mdl.continuous_var_list(range(J), name='b')
    # X：定义客户 i 在站点 j 时间点 t 的流量变量，X[i, j, t] 为连续变量
    X = mdl.continuous_var_dict(x_list_index, name='X')
    # y_list_index：创建一个包含所有时间点和站点的索引组合，用于表示站点是否为“免费时段”的二进制决策变量 Y[t, j]
    y_list_index = [(t, j) for t in range(T) for j in range(J)]
    # Y：定义站点 j 在时间点 t 是否计费的二进制变量，Y[t, j] 为二进制变量
    Y = mdl.binary_var_dict(y_list_index, name='Y')

    # set benders master (0) and SP (1)
    # 为流量变量 X[i, j, t] 添加 Benders 注释，标注其为主问题的一部分。
    for t in range(T):
        for j in range(J):
            for i in range(I):
                X[i, j, t].benders_annotation = 1

    # 为二进制变量 Y[t, j] 和带宽变量 b[j] 添加 Benders 注释，标注其为子问题的一部分。
    for t in range(T):
        for j in range(J):
            Y[t, j].benders_annotation = 0
    for j in range(J):
        b[j].benders_annotation = 0

    # objective
    mdl.minimize(mdl.sum(b))

    # constraints
    for t in range(T):
        # demand
        for i in range(I):
            mdl.add_constraint(mdl.sum(X[i, j, t] for j in range(J) if is_link[j, i]) == demand[t, i])
        # capacity
        for j in range(J):
            mdl.add_constraint(
                mdl.sum(X[i, j, t] for i in range(I) if is_link[j, i]) <= capacity[j])
            mdl.add_constraint(
                mdl.sum(X[i, j, t] for i in range(I) if is_link[j, i]) <= b[j] + capacity[j] * Y[t, j])

    # 免费时段限制：对于每个站点，Y[t, j] 的总和（表示该站点在多少个时间段内是免费时段）不能超过 L。
    for j in range(J):
        mdl.add_constraint(mdl.sum(Y[t, j] for t in range(T)) <= L)

    # mdl.export_as_lp('./experiment_results/MIP_model.lp')



    mdl.solve()
    print('time:', mdl.solve_details.time)
    print('obj: ', mdl.objective_value)

    # if mdl.solve():
    #     print('time:', mdl.solve_details.time)
    #     print('obj: ', mdl.objective_value)



if __name__ == '__main__':
    build_MIP()
